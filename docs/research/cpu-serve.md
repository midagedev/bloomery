# Serving with no card: the executor in the plan, the host engine behind `serve::Engine` (design and paper model)

Round `cpuonly`, 2026-11-04, main `b96cfcc05`. Design only: no repo code changed, nothing run on the box. The three
models' GGUF headers were read by HTTP range requests (header bytes only) from the public uploads `docs/BUILD.md` names;
each total equals that table's size (347,265,635,740 B, 21,155,768,832 B, 111,323,630,080 B). Tags: `[read]` seen in
the code at the cited line and not run; `[measured]` a number some run printed, cited with one copied line;
`[derived]` arithmetic on named inputs; `[assumption]` an input nobody measured; `[estimate]` a size guess for a
change not yet written. Written with AI help (Claude).

The ask (the user, 2026-11-03 and 11-04): "vram 없이 cpu로만 돌리는 모드", expected to fall out of the one placement
that already sizes VRAM → host RAM → NVMe.

## 0. Summary

1. **Zero cards is refused today at four doors, and the plan has no host executor.** The start refuses a missing
   driver (exit 69) before any download, the catalog pick refuses an empty census, `-dev none` is refused as "no seat
   runs on the CPU alone", and a zero-card `Machine` would end in the stage map ("the cards run 0 layers"). The plan
   sizes bytes over three levels, but its executor facts stop at routed stacks (`kernels::host_stack`); the stage
   map (`Stages`) names only cards.
2. **The CPU engine has one whole architecture (deepseek2).** Every main family has its routed stacks on the host
   (`HostLayer`), and every file type has a fused `qdot` kernel. The host lacks the mixers (GQA, delta rule, the V4.1/GLM
   latent attention with its compressor and indexer), the hyper-connections, the routers past softmax top-k, PLE and
   engram glue.
3. **Recommendation: cut (b), with the executor fact as a host stage in the plan's stage map.** One host body over the
   common layer description (`runtime::layer` kinds), served as a `runtime::Target` through the adapter Metal already
   has (`TargetEngine`, lifted to `crates/serve`), planned by `crates/placement` on a zero-card machine. Cut (a), host
   arms in every card body, writes the same host kernels plus an executor branch in each body, and still needs a CUDA
   context to open.
4. **Paper decode `[derived]`, box (5975WX, 256 GB), depth 6:** Qwen3.6-35B-A3B 29 tok/s at today's per-dispatch rates,
   64 at one dispatch per layer (ceiling 75); Qwen3.8 12 → 20 (ceiling 23); V4.1 12 → 17 when its experts were all in
   RAM, but on 256 GB with no card 8.5 % of them page from NVMe: 5.3 tok/s if the misses wait, 9.6 if they are read under
   the layer before. ik at `-ngl 0` measured 9.80 tok/s on V4.1 in a 128-token bench, where every expert it touches fits
   in the page cache: inside the all-resident band, 15 % under our today's-flow point. **64 GB desktop:** Qwen3.6
   8–27, Qwen3.8 2–10, V4.1 about 2 tok/s (most of its bytes come from the drive).
5. **First round:** `bloomery-serve` answers with no card on DeepSeek-V2-Lite (the one model whose every site already
   runs on the host). That round adds the host stage to the plan, `Program::Host` coverage rows, the lifted adapter and
   one host seat. Gates: the existing V2-Lite family gates, plus a served-ids gate against `bloomery-decode` and ik's
   32-prompt argmax, each with FAIL-first.

## 1. What exists

### 1.1 Zero cards today: refused at the start, and incidentally by the plan

| Door | What happens with no card | Where |
|---|---|---|
| Driver | `libcuda` is loaded with `dlopen` (the binary starts on a machine without the driver); a missing library or an old driver is `Driver::Missing`/`Version` | cuda-rust `9df9e7c` `cuda-bindings/src/dyn_load.rs:64` (`"libcuda.so.1"`), `:167` `is_cuda_driver_available`; `crates/gpu-gates/src/start.rs:446-458` `driver_found` |
| Start | "bloomery needs NVIDIA driver R580+ (CUDA 13), and its library libcuda.so.1 did not load" → exit 69 `no-device`, before any download | `crates/serve/src/preflight.rs:283-298`; `crates/gpu-gates/src/bin/bloomery_serve.rs:777-779` |
| Census | an empty census is not an error ("No device is an empty census, not an error") | `crates/gpu/src/lib.rs:2362-2368` |
| Catalog pick | `Machine::read`: "no CUDA device is visible" → exit 69 | `crates/gpu-gates/src/bin/bloomery_serve.rs:961-964` |
| llama spellings | `-dev none`: "no seat runs on the CPU alone"; `-ngl N` below `block_count + 1`: "bloomery runs every layer on its stage card" | `crates/gpu-gates/src/llama_place.rs:866-871`, `:946-957` |
| Plan | no designed case. `plan_rule` builds `Stages::new(model.layers, &machine.cards)`, whose map covers layers by cards only; with no cards it returns `PlacementError::Stages("the cards run 0 layers, the model has N")` `[read]` | `crates/placement/src/placement/planner.rs:235`, `rule.rs:15-41` |

So the plan has three byte levels, but one executor rule. That rule is `kernels::host_stack` ("the host's side of the
executor facts the card's entries above state", `crates/placement/src/kernels.rs:101-118`): the host runs a routed
stack whose type `qdot` fuses. Everything else has an executor because a card stage (`Stages::of_layer`, a card index)
runs its layer.

### 1.2 The CPU engine (`crates/model`)

- **One whole architecture.** `bloomery-decode` runs `RUNS = ["deepseek2"]` (`crates/model/src/bin/bloomery-decode.rs:34`);
  `arch/deepseek2/forward.rs:307` `step` is the only host forward (embed, MLA attention `attn.rs`, dense FFN, MoE, head,
  a `(seq, pos)` KV cache, `kv.rs:76`). Measured: "bloomery 87.87 대 ik 85.87 tok/s(+2.3 % ± 1.0)" at depth 6 (rig-log
  [2026-09-23:284](https://github.com/midagedev/rig-log/blob/main/log/2026-09-23.md)); after a reboot "bloomery 깊이 6
  89.29 / 4096 53.09 tok/s, ik 89.94 / 68.06" (same file, line 279). At depth the host attention is the gap.
- **Shared host modules that every family can use:** `ops::rms_norm` (`ops.rs:350`), the quantized matmuls
  (`ops.rs:611-2102`) over `qdot`, whose fused kernels cover every type in the three files (Q3_K, Q4_K, Q5_K, Q6_K,
  Q8_0, Q5_1, IQ4_NL; `crates/qdot/src/lib.rs:233-257`), `ffn::dense_ffn_with` (`ffn.rs:84`), `head::head_with`
  (`head.rs:56`), `embed::EmbedTable` (`embed.rs:136`), the routed host stacks `moe::HostLayer` (`moe.rs:2694`, built
  from `HostLayerSpec`, `:2196`) and the NVMe tier's seam `moe::TierSlots` (`moe.rs:2543`). `crates/engram` reads row
  tables (engram, PLE) from NVMe with read-ahead.
- **Per-family host views exist only for the routed leg:** `arch/deepseek41/host.rs`, `arch/glm5next/host.rs`,
  `arch/qwen35moe/host.rs` (qwen4exp, qwen35moe, qwen3moe) each wire a family's routed stacks into a `HostLayerSpec`.
  Every family has its facts readers already (`arch/<family>/{hparams,names,spec}.rs`).
- **The host router is softmax top-k only** (`moe.rs:187-195` `route_with`).
- **Host references, not host kernels:** `runtime::hc_gated` holds Qwen3.8's hyper-connection rules and an f64
  reference (`crates/runtime/src/hc_gated.rs:1-20`); `model::mamba2::step_token` is the Mamba-2 host twin a gate reads
  (`crates/gpu-gates/src/bin/gate_mamba2.rs:4`). Neither is a fast path.

What each main family needs that the host lacks, by the `models::Need` vocabulary (`crates/models/src/need.rs:29-215`)
and the header keys read here. The exact list is what `coverage::check` prints against a host program's rows once
they exist (round 1 adds them), so this table is a reading, not a count:

| Site | Host today | V4.1 (`deepseek41`) | GLM-5.3 (`glm5next`) | Qwen3.6 (`qwen35moe`) | Qwen3.8 (`qwen4exp`) |
|---|---|---|---|---|---|
| matmul of the file's types, RMSNorm, head, embed rows | yes | ✓ | ✓ | ✓ | ✓ |
| routed stacks (`HostLayer`) | yes | ✓ | ✓ | ✓ | ✓ |
| router rule | softmax top-k | `Score::SqrtSoftplus` with bias — missing | `Score::Sigmoid` with bias — missing | softmax top-k ✓ | softmax top-k ✓ |
| routed and shared SwiGLU | SwiGLU | with the clamp (`swiglu_clamp_exp`/`_shexp`) — missing | with the clamp (`limit_exp`) — missing | shared expert sigmoid-gated (`Shared{sigmoid_gate}`) — missing | same — missing |
| mixer | `Latent` (V2-Lite MLA) | `Latent` with `q_lora` 1280, `LatentOut::Grouped` (8 groups), sliding window 128, `Compress`/`CompressorApe`, `Selector::StreamTopK` (32 × 128, top 512) — missing | KDA delta-rule layers (`Kind::Kda`) and absorbed `Latent` layers with no rope and a `Selector::TokenPool` index — missing | `DeltaRule` (30 layers) + `Gqa` with `OutGate`, `QkRope` — missing | `DeltaRule` (36) + `Gqa` (12, `compress_ratios` 4, `Selector::TokenPool` 4 × 128, top 2048) — missing |
| recurrent state, conv inputs | none | — | KDA state and conv — missing | `RecurrentState`, `ShortConv` 4 taps — missing | same — missing |
| residual | plain | `Hc` (`HcKind::Mhc`, 4 streams, Sinkhorn 20) — missing | `Hc` (`Mhc`, Sinkhorn) on the trunk — missing | plain ✓ | `HcGated` 4 streams, rank 320 — missing (f64 reference exists) |
| row tables | `crates/engram` rows | `Engram` (layers 1, 14) gate and `engram_wkv` — missing glue | — | — | `Ple` (layer 1, conv 4) — missing glue |

### 1.3 Serving and the host-only loop

- `serve::Engine` (`crates/serve/src/engine.rs:255`) needs `tokenizer`, `prefill` and `next`; the rest (marks, slots,
  drafts) has defaults. Its implementations over a real model are `GpuModel<B>` (`crates/gpu/src/model.rs:3290`),
  `AnyEngine` (`crates/gpu-gates/src/engine.rs:93`, arms `Deepseek2` and `Qwen3moe`, both card), the seats'
  `SeatEngine` (`bind.rs:3484`) and Metal's `TargetEngine<T: runtime::Target>` (`crates/metal/src/server.rs:35-74`).
  `TargetEngine` is host code built on every OS (`crates/metal/src/lib.rs:56-57`); `crates/serve` does not depend on
  `bloomery-runtime` today.
- `crates/runtime` is "host only, no card" (`lib.rs:1`): `Target` (`lib.rs:190`: `prompt`, `prompt_marked`, `step`,
  `keepable`, `cut`, `reset`), `Verify`/`Draft`, the one `generate` loop, the layer kinds a body matches on at load
  (`layer.rs:18-45`: `MixerKind {Gqa, Latent, DeltaRule, Mamba2, ShortConv, None}`, `FfnKind {Dense, Moe}`,
  `ResidualKind {Plain, Hc, …}`), the schedule walk (`sched.rs`), prompt batching (`prompt.rs`), store sizes
  (`stores.rs`), recurrent checkpoints (`seqstate.rs`) and the MTP draft's card-free window (`78237f9d6`).
- `Target` has two implementations over a real model, `crates/app/src/lib.rs:1176` (over a `GpuModel`) and
  `crates/metal/src/arch/qwen3moe/target.rs:83`; the others are test doubles. The host engine would be the third.
- `AnyEngine` refuses `deepseek41`, `glm5next` and others by name before any load (`engine.rs:21-29`); each has its own
  seat in `bloomery-serve` (`crates/gpu-gates/src/bin/bloomery_serve.rs:1-60`). No seat serves deepseek2.
- The NVMe expert tier's arena is host code inside the card crate: `crates/gpu/src/host/nvtier.rs:1-16` (8,547 lines
  with `nvtier/`), "lent to the host leg through the model side's `TierSlots` seam". `crates/model` cannot depend on
  `crates/gpu` (the dependency goes the other way), so a host engine in `crates/model` reaches the arena through the
  seat, which links both.
- The Mac plan already names this cut: [#593](https://github.com/midagedev/bloomery/issues/593) M1, "the CPU engine
  (`crates/model`) behind `serve::Engine`, so `bloomery-serve` answers on a Mac".

## 2. The cut

### 2.1 (a) Host arms for every site inside the card engine's plan

The plan's executor fact becomes per site; every card body routes each site to a card kernel or a host arm, with a
copy across PCIe where two neighbouring sites run on different sides.

- **Owners touched:** `crates/placement` (executor per site), `GpuModel` (`crates/gpu/src/model.rs`: graphs, arena,
  fault word, all CUDA objects; it opens with `Gpu::new()`, `crates/gpu-gates/src/engine.rs:82`), and every body: `Deepseek41Chain`
  (`crates/gpu-deepseek41`, 42,683 lines), `Qwen35Body`/`Qwen38Body` (`crates/gpu/src/arch/qwen3moe`, 46,430),
  `Glm5nextBody` (`crates/gpu-glm5next`, 14,939), then the other bodies of `accepts.rs:17-60`.
- **Lines `[estimate]`:** the host kernels of §2.2 (they are needed either way), plus an executor branch and its
  hand-offs in each body (~1–3k lines a body, 4 main bodies), plus a `GpuModel` that opens without a device.
- **What each model then writes:** a host arm per site in its own body. That is a per-model copy of the shared host
  logic, the shape AGENTS.md "Common code first" forbids.
- **Zero cards:** still no engine until `GpuModel` opens with no CUDA context. A machine without the driver cannot
  open it.
- **What it buys that (b) does not:** sites split between card and host inside a layer. Each hand-off between
  neighbouring sites is a copy each way per token (tens of kB of activation, latency-bound); nothing measured asks for
  that split.

### 2.2 (b) The host engine behind `serve::Engine`, placement deciding RAM vs NVMe

- **Executor fact:** the plan's stage map gains a host stage: `Stages::of_layer[l]` becomes `Card(c) | Host`, `head`
  and `embedding` too. Zero cards means every layer's stage is `Host`. The rule inside a stage stays as it is: a
  stage runs its layers' sites; routed stacks follow the expert rule (card, host resident, or the NVMe tier). The
  always-read set of a host stage must be resident; a host need past the room pages routed experts only, as it does
  today (`docs/serving.md` 5.4).
- **Engine:** one host body over the `runtime::layer` kinds, in `crates/model`. Each kind is one host owner (`Gqa`,
  `Latent`, `DeltaRule`, `ShortConv`, `Mamba2`; `Dense`, `Moe`; `Plain`, `Hc`, `HcGated`). A layer matches on its kind
  once at load. The store sizes come from `runtime::stores`, prompt batching from `runtime::prompt`. It implements
  `runtime::Target`.
- **Serving:** `TargetEngine` moves from `crates/metal/src/server.rs` to `crates/serve` (it is host code, 302 lines with
  its tests; Metal imports it from there). One host seat in `bloomery-serve` opens any architecture whose host
  coverage is complete. An architecture whose coverage is not complete is refused by name (exit 65 `unsupported`),
  with every missing `Need` listed (`coverage::check` against `Program::Host` rows in `AVAILABLE`,
  `crates/model/src/arch/coverage.rs:50`).
- **NVMe:** the seat wires `crates/gpu`'s `NvTier` (anonymous arena, host leg) into the host engine through
  `TierSlots`; nothing is copied. A build without cargo-oxide would need the arena lifted out of `crates/gpu` first
  (question 2).
- **Owners touched:** `crates/placement` (stage map, refusals), `crates/model` (host body and kinds; per-family facts
  already there), `crates/serve` (adapter), `crates/gpu-gates` (seat, start path, `-dev none`/`-ngl 0`). No card body
  and no kernel changes: `just ptx-scan` stays equal for every card bin.
- **Lines `[estimate]`, all common:** stage map and refusals 200–400; coverage rows 100–200; adapter move ~300
  (behaviour-preserving); seat and start path 300–600; host body over the kinds 1,000–1,500; per kind: `Gqa` with
  `QkRope`/`OutGate`/sinks 1,000–1,500, `DeltaRule` + `ShortConv` + `RecurrentState` 1,000–1,500 (GLM's KDA a gate
  variant of it), `Latent` lifted from
  `deepseek2/attn.rs` plus the compressor and indexer variants 2,000–3,500, `Hc` 400, `HcGated` 300, routers 300,
  `Engram`/`Ple` glue 300–500.
- **What each model then writes:** nothing beyond its fact readers, which exist. **What stays per model, by an
  architecture fact:** V4.1's compressor, stream index and engram sites, Qwen3.8's PLE, GLM's KDA gate variant. Each
  is a `Need` that one model's description names. `Hc` (Sinkhorn) serves V4.1 and GLM, the `TokenPool` index GLM and
  Qwen3.8, so those are common kinds from the start. They live in the common kind's module only while two
  models share them, by the same rule as the card side.

### 2.3 Recommendation

**(b), with the executor fact in the stage map.** It keeps one owner per concept:

| Concept | One owner under (b) | Where (a) would duplicate | Where (b) must not duplicate |
|---|---|---|---|
| where bytes live, and which stage runs a layer | `crates/placement` plan (`Stages` with a host stage) | — | Metal planned for itself (`crates/metal/src/arch/qwen3moe/plan.rs`); the host engine takes the common plan, never its own |
| what a layer runs | `runtime::layer` kinds | each card body's own host branch | a second layer walk; the host body walks `runtime::sched` |
| host math per site | `crates/model` kinds | four or more bodies | — |
| a `Target` served as an `Engine` | `crates/serve` adapter (lifted) | — | a second adapter beside Metal's |
| NVMe arena | `crates/gpu` `NvTier` through `TierSlots` | — | a second arena in `crates/model` |
| what a file lacks | `coverage::check` with `Program::Host` rows | — | prose or ad-hoc refusals in the seat |

A later card + host split by layer ranges (llama's `-ngl N` meaning, refused today at `llama_place.rs:946-957`) is
the same host stage inside a card chain. The activation crosses once per stage boundary, as between two cards today.
It reuses the host body, so (b) does not close it off (question 3).

## 3. Paper model `[derived]`

### 3.1 Inputs

| Input | Value | Source |
|---|---|---|
| box DRAM read ceiling | 147.7 GB/s, 32 threads (provisional) | rig-log `docs/bloomery/facts.md:38`: "메모리 읽기 실측 147.7 GB/s(잠정 … 재측정 WKS-35)" |
| box host-tier expert read | 135 GB/s | `docs/HARDWARE.md:20`: "reads 3.37 GB per token at 135–137 GB/s with 32 threads, which is 24.89 ms per token" |
| per-dispatch byte law (CPU engine) | 172 MB → 119, 16.8 → 75, 2–10 MB → 47–51, 0.5–1.1 MB → 13–23 GB/s; kernel MT ceiling 122.6–136.3; pool join 4.65 µs | rig-log `facts.md:108`: "사이트 달성 GB/s는 디스패치당 바이트의 단조 포화 함수(172MB→119, 16.8→75, 2–10MB→47–51, 0.5–1.1MB→13–23 GB/s). 커널 MT 상한 122.6–136.3 GB/s" |
| Qwen3.8 host expert | 23.6 µs a slot (3.134 MB) | `docs/research/tierengine.md:482`: "480 × 23.6 µs = 11.3 ms" |
| box NVMe (nvme0n1, Gen4 ×4) | QD 1 / 2 / 4 / 8 / 16: 2.513 / 4.153 / 5.666 / 6.482 / 6.769 GB/s | rig-log [2026-10-11#qdsweep](https://github.com/midagedev/rig-log/blob/main/log/2026-10-11.md#qdsweep): "\| 16 \| 6.769 [6.425..6.987] \|" |
| Gen5 drive (a Spark's) | expert parts at QD 16 8.4–8.9 GB/s | `docs/contrib/dgx-spark.md:531-532`: "the two drives read 8.4 to 8.9 and 5.8 GB/s" |
| engram rows from NVMe | 0.31 ms a token with read-ahead | `docs/HARDWARE.md:26`: "with each row's read-ahead issued one token early, 0.31 ms" |
| host MAC rate (union GEMM) | 1.31 TMAC/s at c = 27.0 µs; 2.42 at c = 14.63 µs (35.39 M MAC a column-expert) | rig-log `plan.md:104` ("35.39 M MAC … → 1.31 TMAC/s"); [09-25#h3tile-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#h3tile-ab): "\| unionr8(큐 + row-lane) \| 28.23 \| 46.96 \| 59.4 · 14.63 \|" |
| box memory | 8 × DDR4-3600 = 230.4 GB/s theoretical; 147.7 is 64 % of it | `docs/HARDWARE.md:10`; [derived] |
| box room for the model | 240 GB of 251 GB visible | [assumption]; the plan reads it (`host_available_read`) |
| 64 GB desktop DRAM | 33–57 GB/s (dual-channel DDR4-3200 … DDR5-5600 at the box's 64 %) | [assumption]; room 56 GB [assumption] |
| desktop MAC rate | 0.33–1.2 TMAC/s (8–16 cores at the box's rate per core) | [assumption] |

### 3.2 Bytes a token, from the headers

Routed bytes a token = Σ routed stacks × k / E; always-read = every other matrix, the norms, the shared experts and
the head; `token_embd` and the row tables are a few rows a token.

| Model (file) | k / E, layers | routed total | routed a token | always-read a token | row tables (rows a token) | active a token | weight MACs a token |
|---|---|---|---|---|---|---|---|
| V4.1 `Q3_K_M` (347.27 GB) | 6 / 384, 40 | 258.768 GB | 4.0432 GB | 3.7311 GB (3.5960 without `engram_wkv`) | engram 84.48 GB (48 × 110 B) | **7.774 GB** | 16.13 G |
| Qwen3.6-35B-A3B `Q4_K_M` (21.16 GB) | 8 / 256, 40 | 19.504 GB | 0.6095 GB | 1.3662 GB | — | **1.976 GB** | 2.95 G |
| Qwen3.8 `UD-Q4_K_XL` (111.32 GB) | 10 / 512, 48 | 77.018 GB | 1.5043 GB | 4.8302 GB (Q8_0 trunk) | PLE 28.80 GB (16 × 90 B) | **6.334 GB** | 6.67 G |

Cross-checks: V4.1's always-read without the engram projections, 3,595,952,960 B, is within 35 kB of facts.md's
3,595,917,760 B ("밀집(토큰마다 읽는 것, `token_embd`·engram 제외) 3,595,917,760 B", rig-log `facts.md:18`). The routed 4.043 GB
equals `v41-inventory.md`'s 4,043,243,520; that file's "dense 7,657,593,280" is the Q8_0-attention set, not this one.

### 3.3 Decode

    t_token = A / B_dense + h · R / B_routed + S / B_routed            DRAM (cores stream it: one resource at M = 1)
            + (1 − h) · R / B_nvme(QD)                                  NVMe misses; QD ≈ (1 − h) · k · 3 parts a layer
    wall    = DRAM + NVMe while a miss waits for its read; max(DRAM, NVMe) once the read runs under the layer before
    h       = min(1, (room − A − KV) / routed total)                    uniform routing: the floor; residency raises it

`A` always-read, `R` routed a token, `S` recurrent state and KV traffic (Qwen3.6 126 MB, Qwen3.8 226 MB at depth 6,
read and written), KV = 2 GB [assumption]. At M = 1 the `qdot` kernels run at the machine's ceiling (122.6–136.3 GB/s
against 147.7), so host compute stays under the DRAM stream; the term that moves is the dispatch, not the kernel.

**Box (5975WX, 256 GB), depth 6:**

| Model | h | today's flow: one dispatch a matrix (law) | one dispatch a layer (A at 122.6) | NVMe misses | tok/s |
|---|---|---|---|---|---|
| Qwen3.6 | 1 | 381 dispatches, A at 53 GB/s 26.0 ms + R 8.7 ms → **28.9 tok/s** | 11.1 + 4.5 → **64 tok/s** | 0 | ceiling 74.8 |
| Qwen3.8 | 1 | 774 dispatches, A at 66 GB/s 73.0 ms + R 11.3 ms → **11.9 tok/s** | 39.4 + 11.3 → **19.7 tok/s** | 0 | ceiling 23.3 |
| V4.1, all experts in RAM | 1 | A 56.4 ms + R at 135 GB/s 29.9 ms → **11.6 tok/s** | 30.4 + 29.9 → **16.6 tok/s** | 0 | ceiling 19.0 |
| V4.1, no card (256 GB) | 0.905 | DRAM 56.4 + 27.1 = 83.5 ms | 30.4 + 27.1 = 57.5 ms | 0.383 GB at QD ≈ 2, 3.67 GB/s → 104 ms | **5.3** (wait) / **9.6** (read under the layer before) |

The law's rates are sites' achieved rates, so the pool's join (4.65 µs, 421 a token on Qwen3.6) is inside them, not
added. The wide band the law's spread gives (every trunk matrix at 47 GB/s and the experts at 75, up to the bulk
column) is 7.5–16.6 tok/s on V4.1 all in RAM, 25.7–60.3 on Qwen3.6, 7.9–19.2 on Qwen3.8.

**Sanity check.** ik at `-ngl 0` on V4.1: "`-ngl 0` `llama-bench` tg64 2반복에서 **ik 디코드 9.80 tok/s, mainline 7.98**"
(rig-log [2026-09-13:31](https://github.com/midagedev/rig-log/blob/main/log/2026-09-13.md); the line does not name which of
the two Q3_K_M sets ran). In 128 random-token steps a layer touches 384 · (1 − (1 − 6/384)^128) ≈ 332 experts
[derived]. That is 86 % of the routed bytes, about 229 GB with the trunk, which fits the page cache, so the bench is
the all-resident row. 9.80 tok/s is 102 ms, 76 GB/s on 7.774 GB. That is inside the band (7.5–16.6) and 15 % under
today's-flow point (11.6). ik's expert gather from the mapping read 79.8 GB/s on this box ("3.44 GB/token ÷ 43.1 ms =
79.8 GB/s", rig-log `docs/throughput-model.md:34`), against the host tier's 135. A second check, GLM-5.3 at `-ngl 0`:
"8.3 tok/s로 145 GB/s·토큰당 13.4 GB 산수와 맞는다" (rig-log [2026-09-15:38](https://github.com/midagedev/rig-log/blob/main/log/2026-09-15.md)),
111 GB/s on a Q8_0 trunk of large matrices: the law's large-dispatch end.

**64 GB desktop** (DRAM 33–57 GB/s, the box's drive curve; a Gen5 drive in brackets):

| Model | h | DRAM | NVMe | tok/s, misses wait | tok/s, reads overlapped |
|---|---|---|---|---|---|
| Qwen3.6 | 1 | 36.9–127 ms | 0 | 7.9–27.1 | same |
| Qwen3.8 | 0.638 | 106–365 ms | 0.544 GB at QD ≈ 11: 82.6 ms [64.1] | 2.2–5.3 [2.3–5.9] | 2.7–9.5 |
| V4.1 | 0.194 | 79–274 ms | 3.258 GB at QD ≈ 15: 485 ms [379] | 1.3–1.8 [1.5–2.2] | 2.1 [2.6] |

On the desktop V4.1 is a drive stream, about 2 tok/s. Qwen3.6 is the model a 64 GB machine serves well.

### 3.4 Prefill

At P ≥ 512 every V4.1 expert is touched per ubatch (384 · (1 − (1 − 6/384)^512) ≈ 383.9 a layer [derived]), so the
ubatch reads the whole routed set once: 258.8 GB / 135 GB/s = 1.9 s. The MACs take 8.26 TMAC / (1.31–2.42 TMAC/s) =
3.4–6.3 s. **Prefill is compute-bound on AVX2**, with DRAM under the cores' shadow:

| Model | weight MACs, P = 512 | box (1.31–2.42 TMAC/s) | desktop (0.33–1.2) | note |
|---|---|---|---|---|
| V4.1 | 8.26 TMAC | **81–150 pp tok/s**; with the union fit (a + c·m̄ = 59.4 + 14.63 × 8 µs an expert) for the routed part, 90–119 | 20–74, but the 209 GB paged set per ubatch streams at 6.77 GB/s = 31 s: **≤ 16.5** | the box's 24.6 GB paged set streams in 3.6 s, under the compute when read in file order beside it |
| Qwen3.6 | 1.51 TMAC | **445–821** | 112–407 | the delta-rule scan and attention are not in the count; P = 4096 adds ~6 % attention MACs [derived: 10 layers × 16 heads × 256 × 2 × 2048 mean keys] |
| Qwen3.8 | 3.41 TMAC | **196–363** | 49–180 | Q8_0 trunk GEMM rate on the host unmeasured |

On a paged plan a prompt's stream is fixed per ubatch, so the ubatch should be the whole prompt (bulk).

### 3.5 Resource timeline (one decode token)

| Case | cores | DRAM | NVMe | PCIe | wall |
|---|---|---|---|---|---|
| Qwen3.6, box | busy streaming | 1.976 GB: 13.4 ms at the ceiling, 34.7 ms by the law | 0 | 0 | DRAM + 421 pool joins: the joins and small dispatches are the gap |
| Qwen3.8, box | busy streaming | 6.334 GB: 42.9 ms at the ceiling, 84.3 by the law | 0 | 0 | same; the Q8_0 trunk is 76 % of the bytes |
| V4.1, box, no card | streaming 57.5–83.5 ms, then idle while a miss waits | 7.39 GB | 0.383 GB, 104 ms at QD ≈ 2 | 0 | sum today; max once each layer's misses are read during the layer before |
| V4.1, 64 GB | mostly idle | 0.98 GB + A | 3.26 GB, 379–485 ms | 0 | the drive |

The step's cycles against every unit's demand: on Qwen3.6 the DRAM demand at the ceiling is 13.4 ms against a 34.7 ms
step, and the instruction demand of a GEMV at M = 1 is under the DRAM (the kernel's MT ceiling is the DRAM's). So the
step is latency the threads do not hide, and a faster kernel predicts 0. The levers, in order:

1. **Bulk:** one pool dispatch per layer's projections instead of one per matrix (the law's 16.8 MB → 75 GB/s against
   2–10 MB → 47–51 is the whole gap). Worth 34.7 → 15.6 ms on Qwen3.6, 84.3 → 50.7 on Qwen3.8 [derived].
2. **Asynchrony:** on a paged plan, read layer l+1's misses during layer l. The router's lookahead (the next
   router on this layer's residual, `docs/research/tierengine.md` (c)) gives the ids. Worth min(DRAM, NVMe) a token:
   V4.1 no-card 5.3 → 9.6 tok/s.
3. **SIMD** is already in place (`qdot` AVX2 at the ceiling). AVX-512/VNNI ([#81](https://github.com/midagedev/bloomery/issues/81))
   moves prefill on Zen 4/5 desktops, not decode.
4. **SIMT** has no host analogue at M = 1; prefill already runs the union GEMM over the whole ubatch.

### 3.6 The one term derivation cannot fix

The host engine's achieved rate on the always-read set at these shapes. The law was measured on V2-Lite's matrices in
September; Qwen3.6's are other sizes and its dispatches are not written yet. Predicted 53 GB/s at one dispatch a
matrix (28.9 tok/s), 122.6 at one a layer (64 tok/s). The run belongs to the round that first serves Qwen3.6 (round 2),
on its own card: term "Qwen3.6 host decode tok/s @ depth 6, the box, no card used"; band 25–34 tok/s at one dispatch
a matrix [derived: the trunk between 47 GB/s and the law's 53, the experts between the law's 70 and the tier's 135];
condition: the lease and witness blocks, no card process; next action: inside the band, open the bulk dispatch round
with 64 tok/s as its prediction; below 25, `BLOOMERY_PROFILE=1` level 1 per site before any kernel change; above 34,
the law is pessimistic for these shapes, and the bulk round's prediction is rescaled from the measured rate.

## 4. First round: `bloomery-serve` answers with no card on DeepSeek-V2-Lite

Why V2-Lite: it is the only architecture whose every site already runs on the host, with gates against ik
(`gate-ds2-debug`, `gate-ds2-release`: the per-layer bands `tests/forward.rs:97` `L_OUT_BANDS`, logits
`LOGIT_REL_TOL 5e-2` at `:106`, the 32-prompt argmax against ik with `KNOWN_DIVERGENCE = []` at `tests/prompts.rs:209`).
So the round moves no numeric path; it lands the plan, the coverage rows, the adapter and the seat that every later
model reuses.

Scope:
1. `crates/placement`: a host stage in `Stages` (`rule.rs:15-41`); a zero-card `Machine` plans every layer, the head
   and the embedding on it; the host need counts the always-read set as resident and refuses by name a set that does
   not fit; routed stacks keep the expert rule. Pure crate: its tests run on the Mac (`just mac-test`).
2. `coverage.rs`: `Program::Host` and its `AVAILABLE` rows for what `crates/model` runs (the deepseek2 sites, the
   routed stacks, matmul types by `qdot::supports`). `coverage::check` against it lists by name what any other file
   lacks.
3. `crates/serve`: `TargetEngine` moved from `crates/metal/src/server.rs` (with its tests), and `bloomery-runtime` added
   as a dependency; `crates/metal` imports it. A move: Metal's own tests unchanged.
4. `crates/model`: `runtime::Target` for the deepseek2 host forward (`forward.rs:307` `step`, `new_cache`); `cut` keeps
   only what `KvCache` can take back (truncation added there, or `keepable` = the current position until it is).
5. `crates/gpu-gates`: the host seat. A missing driver (`Driver::Missing`), an empty census, `-dev none` or `-ngl 0`
   plans on the host instead of refusing. A file whose host coverage is incomplete ends 65 `unsupported` with the
   list. A file a card seat serves keeps its card seat whenever a card is visible and no flag asks for the host.

The plan record on the box under `CUDA_VISIBLE_DEVICES=""`, V2-Lite (fields of `bloomery.plan/1`, `start.rs:1-9`):
`seat host`, `place host` (stage map: host, layers 0..27, head), `need {cards: [], host_bytes: <the file's resident
bytes>, paged: 0}`, and one human line `plan stage host layers 0..27 head host` beside the existing `need` line. Qwen3.6 on the
same start: exit 65, listing its `DeltaRule`, `Gqa`/`OutGate`, `ShortConv`, `RecurrentState` and shared-gate items.

Gates, each with FAIL-first:
- `just mac-test` (placement): a zero-card plan puts every layer on the host stage; a host need past the room is
  refused by name. FAIL-first: remove the host arm from `Stages::new`, and the old `stage map: the cards run 0 layers`
  refusal turns it red.
- `gate-ds2-release` and `gate-ds2-debug` unchanged (family deepseek2, `tools/gate-class.tsv:14,34`).
- New `gate-host-serve` (class common, owners: the host seat and the plan's host stage; its row in
  `tools/gate-class.tsv` and `tools/owners.tsv`): start `bloomery-serve` with no visible card on V2-Lite; the server's
  greedy ids for the 32 prompt sets equal `bloomery-decode`'s bit for bit, and their first argmaxes equal ik's
  (`KNOWN_DIVERGENCE` stays empty). FAIL-first: the adapter starting `step` one position late (one line) turns the
  ids red; reverted, green.
- `just ptx-scan` equal to base for every card bin (no kernel moves). The class is add: new host entries, no engine
  caller of the card path changes.

No timed run: the round's decision is functional, and V2-Lite's tok/s is measured (89.29 at depth 6). The wall of
the new gate has no `gate-times.tsv` row; ≤ 15 min [estimate], an express landing.

Rounds after it, each a landable step with its own model:
- **R2 Qwen3.6** (fits 64 GB): the host body over the kinds, `Gqa` + `QkRope` + `OutGate`, `DeltaRule` + `ShortConv` +
  `RecurrentState`, the sigmoid shared gate; `crates/model/RESULTS.md` rows for the new kinds against ik's CPU dumps;
  the §3.6 run.
- **R3 bulk dispatch** (per-layer pool dispatch of the always-read set), if R2's run lands in its band.
- **R4 Qwen3.8:** `HcGated`, PLE glue, the compressed GQA layers and their index. The NVMe tier through `TierSlots`
  for a 64 GB machine.
- **R5 V4.1:** `Hc` (Sinkhorn), the `Latent` kind lifted from deepseek2 with the compressor, index and grouped output
  LoRA, the sqrt-softplus router, engram glue, and the lookahead read for misses (§3.5 lever 2).

## 5. Open questions for the user

1. **Which model goes first?** Recommendation: V2-Lite first (one round, existing gates, no numeric change), then
   Qwen3.6 as the first model worth using without a card (29–64 tok/s on the box, 8–27 on a 64 GB desktop [derived]).
   The other way, Qwen3.6 first, puts the plumbing and two new mixer kinds in one larger round.
2. **One binary, or a CPU-only build?** Recommendation: the same `bloomery-serve`. `libcuda` is opened with `dlopen`,
   so the release binary already starts without the driver; only the refusals of §1.1 change. A plain-cargo CPU
   build (no cargo-oxide, for non-NVIDIA Linux or the Mac's #593 M1) needs the host tier and `NvTier` lifted out of
   `crates/gpu` first. Do that when someone asks for it.
3. **Card + host split by layers (llama's `-ngl N`)?** Recommendation: later. The host stage is shaped so a card
   chain can call it, but no round opens until a measured case needs it, for example a card that cannot hold the dense
   trunk. Until then `-ngl N` keeps its named refusal and `-ngl 0` means the host stage.

## Appendix: how the header numbers were read

A scratch Python reader (not in the repo) fetched each shard's header with growing `Range: bytes=0-N` requests from
`https://huggingface.co/{vcruz305/DeepSeek-V4.1-Flash-GGUF, lmstudio-community/Qwen3.6-35B-A3B-GGUF,
unsloth/Qwen3.8-Flash-Next-GGUF/…/UD-Q4_K_XL}` and sized each tensor by its type's block size and bytes per block
(the table `tools/ref/gguf-ranges.py` uses). Routed means a name with `_exps`; row tables are `engram_embd` and
`per_layer_token_embd`; `token_embd` is one row a token; every other tensor is read each token. Header keys used:
`expert_count`, `expert_used_count`, `block_count`, `full_attention_interval`, `compress_ratios`, `engram.layer_ids`,
`ple.layers`, `hyper_connection.*`.
