# Optimization axes

How a bloomery performance change is classified before it is queued. A round names one axis, the hot path it sits on, the instrument that shows headroom on that axis, and a pass/fail derived from code and numbers already measured. A GPU run confirms that prediction. It does not search.

Tags: `[measured]` a printed number with its source; `[derived]` arithmetic on those numbers or on code; `[assumption]` a stated input that is not a measurement; `[stated]` a figure copied from a source this round did not re-open. A struck value is one a later measurement replaced. The GPU call sites in gap 1 were read at `d90151ed`. `origin/main` `bd29e9c9` does not change those files: it changes `crates/qdot`'s f16 scale load and a mac-remote self-test stub. `dot_row_cols` still unpacks one weight row for 1..=`TILE_COLS` columns.

The inventory this file classifies is a maintainer's optimization inventory (lineage `9bd7d577`) and a harvest of role notes that are not in this repository. A figure cited as `harvest` is `[stated]`: its page is not here to re-open. This file does not rank work and does not take tier/LES, quant formats, or `q35hybpp`; other tracks own them.

## References

The seven axes below are bloomery's framing. Each one is the form of a standard model:

- **Roofline.** Samuel Williams, Andrew Waterman, David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *Communications of the ACM* 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>. The full model is EECS-2008-134, <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html>. Attainable performance is the minimum of the machine's peak flop rate and the product of peak bandwidth and arithmetic intensity. The ridge point is the intensity where those two meet. Axis A asks whether a kernel sits on the bandwidth roof. Axis D asks whether doing more with the same bytes moves the working point toward the ridge, where extra arithmetic is free.
- **Bentley, efficient programs.** Jon Louis Bentley, *Writing Efficient Programs*, Prentice-Hall, 1982. Bibliographic record: <https://dl.acm.org/doi/10.5555/578475>. The rules that match these axes are code motion and procedure collapse (fixed work paid once: D and G), loop fusion (D), caching and "do the common case fast" and exploiting algebraic identities (E), and laying data out so the next consumer hits (F). A rule that does not name which term of the step it shortens is not a round.
- **USE.** Brendan Gregg, "The USE Method", <https://www.brendangregg.com/usemethod.html>, and *Systems Performance*, 2nd ed., Addison-Wesley, 2020, <https://www.brendangregg.com/sysperfbook.html>. For each resource, check utilization, saturation, and errors. Utilization against a known peak is axis A. Saturation (a queue) while another resource is idle is axis B or C. Errors here are faults and re-reads (`majflt`, mapping faults, rereads), which are axis E or F. A resource at low utilization is not the bound.
- **Little's law and GPU latency hiding.** John D. C. Little, "A Proof for the Queuing Formula: L = λW", *Operations Research* 9(3), 1961, <https://doi.org/10.1287/opre.9.3.383>. In the form bloomery uses (`docs/research/sparkdec.md` §5): bytes in flight per SM must be at least bandwidth × latency / SM count, or the warps do not cover the memory round trip. Vasily Volkov, "Better Performance at Lower Occupancy", GTC 2010, <https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf>, is the GPU statement that occupancy is not that quantity: a kernel can be full of warps and still stall, or hold few warps and still hide latency, depending on bytes outstanding. The CUDA C++ Best Practices Guide states the same occupancy limit: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy>. Axis C is this inequality. When it already holds, the model says 0.
- **Software prefetch**, for axis B's prefetch half. David Callahan, Ken Kennedy, Allan Porterfield, "Software Prefetching", ASPLOS IV, 1991, <https://doi.org/10.1145/106972.106979>. A prefetch pays only when the miss would have sat on the critical path and the prefetch's own work fits under other useful work.

Two rules from `AGENTS.md` apply to every axis. Draw the resource timeline first: per unit of work, how long the host CPU, the card SMs, PCIe, host DRAM and NVMe are busy, and whether the wall is their **sum** or one resource's **max**. A term under another resource's shadow is worth 0. And the ruler: same-binary decode scatters with SD 0.6 %, so the 95 % interval on a difference of two arm means is ±1.0 % at four rounds and ±0.8 % at six (`AGENTS.md`, Measuring). A predicted effect inside that band is not a round. `lease_take` refuses a timed run that has no prediction card (`docs/cards/<slug>.card`, `tools/ref/card.py`).

`BLOOMERY_STEP_STATS=1` slows a step by 2.5–3 ms (`docs/BUILD.md`). Compare an instrumented run only with an instrumented run. `ncu` serializes kernels and locks the clock; its microseconds are not a timing record (`tools/ref/ncu-gpu.sh`). `nsys` leaves the run in place and is the timeline (`tools/ref/nsys-gpu.sh`).

## The axes

### A. Bandwidth to the ceiling

**Question.** Of the bytes the critical path actually reads, what fraction of the plain-read rate of that resource do they move at?

The roof is Williams' bandwidth roof: time ≥ bytes / peak bandwidth. Bloomery's roofs are measured plain reads, not spec peaks. Examples already used: GB10 plain read 235.6 GB/s (`sparkdec.md` §1, `#13` `hostread2`); A6000 `_sel` kernels at 579–701 GB/s `[stated, harvest synth §2, citing rig-log plan.md:73; that page was not re-read here]`; the host leg at 130 GB/s of a 148 GB/s machine (`docs/performance.md`, the V4.1 host-leg bench). `docs/roofline.md` is the byte side of the same model: active bytes per token, not file size, because an unrouted expert is not read.

**Headroom.** `(plain_rate − achieved) / plain_rate`, and only when the kernel is memory-bound. A kernel whose SM pipes are the busy ones is on the other side of the ridge; speeding its loads predicts 0. Instruction-only edits on a kernel near ~700 GB/s predict 0 (`AGENTS.md`).

**Instruments.**

| What | Where |
|---|---|
| Which kernel holds the step, and the gap between the kernel sum and the window wall | `tools/ref/nsys-gpu.sh`. The window wall has to match `depth-gpu` / `time step` before the table sits next to a record |
| Memory-bound vs compute-bound, DRAM % of peak, bytes read | `ncu` sections `SpeedOfLight` and `MemoryWorkloadAnalysis`; metrics `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes_read.sum` (`ncu-gpu.sh`, depth and ds41pp/gemm forms) |
| Tensor pipe, for the other side of the ridge | `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` (gemm form) |
| Host-leg time vs bytes | `stat step` `leg_us` under `BLOOMERY_STEP_STATS` (`record.rs` `STAT_STEP` / `STAT_STEP_HOST`) |
| NVMe rate vs the drive knee | `nvread` (`median` / `p10` / `p90` GB/s, `threads`, `bytes`) |
| Expert-stream copy rate | `xstream end` `lane_gbs` |
| Active bytes, the roofline denominator | `docs/roofline.md`; a new file is re-counted with `gguf-inventory`, not estimated |

### B. Overlap waiting

**Question.** Is the wall a sum of resources that could be a max? A wait on the critical path, with another resource already holding the bytes or the result, is the headroom.

This is pipelining and prefetch. Callahan's prefetch and the tierengine in-layer overlap (`experts_into_on` computes hits while the miss is in flight, tierengine S3) are the same shape. A prefetch of something the LRU cannot have evicted is worth 0 (tierengine §3(c), signal 1). A term in shadow is worth 0: GLM's card-shared expert is 1.6–1.9 ms under the bridge `[stated, harvest §2, citing glmops-design.md]`.

**Headroom.** The uncovered wait: host wait not overlapped with card work, card gap not overlapped with the host leg, a fill whose `fill_ns` sits in series with the hits' compute.

**Instruments.**

| What | Where |
|---|---|
| V4.1 prompt: host wait on the route's copies, vs card time | `stat prefill split` `wait_ms`, `wait_lb`, `card_out_ms`, `card_in_ms`; per layer-batch `stat prefill lb` `wait_ms`, `card_out_ms`, `card_in_ms`. The landed overlap is the host wait 17.8 → 0.8 ms per layer-batch `[measured, docs/performance.md, rig-log 2026-09-26#prefillgroup-ab]` |
| Qwen/GLM prompt walks | `stat prompt lb` `wait_ms`, `card_down_ms`, `card_shadow_ms`, `card_gap_ms` (stream end to the other unit's first launch); sums in `stat prompt split` |
| Decode: card idle from the host's signal to the next go | `stat step` `gap_min_us`, `gap_p50_us`, `gap_max_us`; `go_early`, `overlap`, `leg_us` |
| Residency copies that missed the boundary | `residency pass` `late`, `wait_us`, `stage_us`, `in_flight` |
| Fill wall beside the step | `nvtier` `fill_ns`, `step_fills`, `step_hits`; `tier census` `dispatch_wait_ns` |
| Staging backlog on the prompt's pick | `call stream` `backlog_us`; `xstream` `backlog_us` |
| A cached state's copy the request waited on | `cache landed` `waited_ms`; `cache load` `waited_ms` |
| Engram read vs the step | `stat step` `eng_wait_us`, `eng_cold`, `eng_warm` |

`nsys` shows the same fact as a gap between kernels on one stream. PDL (programmatic dependent launch) is the device-side attempt to overlap kernel n's tail with kernel n+1's loads. It was measured slower on GB10 on top of the row prefetch (`sparkdec.md` §10) and is closed.

### C. More in flight

**Question.** Does the busy resource have enough outstanding work to cover its latency? Little's law: need = bandwidth × latency. For a GPU SM that is bytes in one round trip. For NVMe it is queue depth × request size. For a link it is messages in flight × message size.

Volkov's point is the guard: raising occupancy, or adding warps, predicts 0 when the bytes already in flight meet the need. `sparkdec.md` §5.3 applies it to GB10 decode gemvs and gets 0 for the "more warps" form. The row-locality form, which is axis F, was the term that paid.

**Headroom.** `long_scoreboard` (waiting on a global load) is a large share of warp stalls, and the bytes one round trip carries are under the need. `not_selected` being large means the scheduler already had another warp ready: occupancy is not the shortage.

**Instruments.**

| What | Where |
|---|---|
| Stall reason | `ncu` `WarpStateStats`; the named stalls in `ncu-gpu.sh` (`long_scoreboard`, `barrier`, `short_scoreboard`, `mio_throttle`, `not_selected`, `no_instruction`) |
| Occupancy and what caps it | `ncu` section `Occupancy` (registers, shared memory, block count) |
| Bytes one warp has outstanding | the kernel's load schedule, as `sparkdec.md` §5.2 did from PTX order. SASS not read is `[unverified]` |
| DRAM latency the need is multiplied by | a measured latency, or an explicit `[assumption]`. GB10's L in `sparkdec.md` §5.1 is an assumption band (0.75–1.5 µs), not a GPU-side measurement |
| Drive queue depth | `nvread` `threads` and `ms`; the tier reader's per-kind waits. S0's `KindStats.run_ns` and the fault/read split are still not in main (`tierengine.md` §5, S0) |
| Copies the residency machine has outstanding | `residency pass` `in_flight`, `bytes` |
| One graph replay vs an eager launch chain | `nsys` replay count; `capture` record; `cnode_probe` for the empty-node price |

### D. More at once

**Question.** A byte read, or a fixed cost, divided by how many rows share it. Below the roofline ridge, extra columns on the same weights are free until arithmetic intensity reaches the ridge. `docs/roofline.md` derives that ridge for the host Q3_K kernel: the measured k* is about 2.2 columns `[measured, rig-log 2026-09-21]`, not the earlier flop-only 5.4. Fusion is the same axis when it removes a launch whose work the neighbour can absorb (`AGENTS.md`: Δt = −ΔN × c_node − the removed kernel's time + the work every block now repeats).

**Headroom.** `pass(m) / pass(1)` above 1, on a kernel the roofline says is bandwidth-bound at m = 1, and the implementation reads the weight once. If the implementation re-reads per column, the ratio stays near m and the axis is open. If the kernel is already compute-bound at m = 1, extra columns cost full price.

**Instruments.**

| What | Where |
|---|---|
| Rows a prompt pass actually runs | `time prompt` `n`, `passes`; `stat prompt lb` `cols` |
| Whether the union is wider than one expert's reuse | `stat prompt lb` `experts`, `m_max`, `m_hot`, `cols_hot`, `m_sq` (spread). A large `experts` at small `m_max` means the batch did not share weights |
| Positions per draft pass | `draft summary` `positions`, `passes`, `tok/s(positions)`; `mtp summary` `kept`, `widths`, `realized_e`, `predicted_e`; `time pass` `positions` |
| Launch once per row vs once per pass | `nsys` launch count of the expert kernel inside one `time prompt` or one verify pass |
| The code contract | host `qdot::dot_row_cols` unpacks a row once for 1..=`TILE_COLS` (8) columns (`crates/qdot/src/lib.rs`); card grouped tiles read each weight row once for up to eight slots (`crates/gpu/src/kquant/tiles.rs`); the wide arm reads each weight once per unit when `m > GEMV_COLS` (`crates/gpu/src/arch/qwen3moe/wide.rs`) |

### E. Less work

**Question.** Fewer bytes or fewer operations on the critical path: a smaller quant, a speculation that removes target passes, a cache that does not re-read, a sparsity pattern that does not load the cold expert. Bentley's caching and common-case rules live here. The roofline says what the removed bytes are worth: bytes / achieved rate, and only if those bytes were on the wall.

A speculation's gain is positions per target pass (`draft summary`, `mtp summary`), not the draft's own FLOPs. A draft that re-reads the target's expert bytes per row pays that tax and can lose the acceptance (`docs/roofline.md`: a 4-token verify measured at 1.69× one token's pass, 2026-09-19, before the grouped path). That measurement is historical. It is not today's expected gain.

A quant that does not reduce bytes predicts 0. NVFP4 at 4.5 bpw equals Q4_K (144 B / 256) `[derived, harvest §5, quant.rs Q4_K block 144]`. W4A16-style activation schemes were rejected because M = 1 is already bandwidth-bound and the weight bytes do not fall (`quant-decode-efficiency.md` decision table).

**Headroom.** Critical-path bytes that a smaller representation, a hit, or a skipped pass would not move, priced at the achieved rate of the resource that moves them.

**Instruments.**

| What | Where |
|---|---|
| Active bytes per token | `docs/roofline.md`; indexer bytes grow with depth (the attention term does not, on V4.1) |
| Where a routed pick was served | `tier census` `card`, `arena`, `file` (the three sum to `picks` when each host slot is read once) |
| Fills vs hits | `nvtier` `step_fills`, `step_hits`, `misses`, `mapping_reads`, `buffered_reads` |
| Page faults and engram cold rows | `stat step` `majflt`, `minflt`, `eng_cold`, `eng_warm`, `eng_direct` |
| Experts read again because the host page was gone | `residency pass` `rereads`, `reread_bytes`, `faulting`, `unresident` |
| Prompt-cache hits | `cache reuse` `ask` vs `kept`; `cache load` `common`, `kept`; `cache skip` `why` |
| Draft acceptance | `draft summary` `accepts` / `proposals`; `mtp window` `accepted` vs `width`; `mtp summary` `realized_e` |

### F. Locality and placement

**Question.** The byte is read from the tier that was supposed to hold it, and the next consumer finds it there. This is not "fewer bytes" (E) and not "higher bandwidth" (A). A hit on the wrong tier can still be the right number of bytes and a low rate: DRAM row locality on GB10, a mapping fault instead of an arena slot, an expert that should have stayed on the card.

The landed GB10 prefetch is this axis. L2 read sectors were already 1.01–1.05× the memory fills (`sparkdec.md` §1), so the kernel was not re-reading through L2; the deficit against 235.6 GB/s was row locality, and a per-warp L2 row prefetch cut the step 11.439 → 9.889 ms `[measured, sparkdec.md §10]`. The same prefetch on the A6000 was 0.957× `[measured, same section]`. Little's-law occupancy (axis C) was the form that predicted 0.

**Headroom.** Hit rate below what the slot book can hold, or a sector amplification well above 1, or a `file` / `mapping_reads` count on ids the arena was built to serve.

**Instruments.**

| What | Where |
|---|---|
| L1/L2 hit rate and sector amplification | `ncu` `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`; the gemm form's source page, L2 theoretical sectors vs ideal |
| Arena hits, evictions, resident bytes vs budget | `nvtier` `step_hits`, `step_fills`, `evictions`, `resident_bytes`, `budget`, `keep_claims`, `transient_claims` |
| Card vs arena vs file | `tier census` |
| Promotions that landed late or not at all | `residency pass` `landed`, `late`, `made`, `rereads`; `residency unset` for why the machine is off |
| Prompt pick vs the host union | `call stream` `admitted`, `kept`, `admit_cols`, `victim_cols` |
| Plan | `plan` record (which tier, arena bytes) |

Router-locality replay (LRU-64 hit rate 67.7 % prose / 73.3 % code) is `[measured, docs/contrib/router-locality-20261012.md]`, and it is the premise of tierengine S5, which unitier owns.

### G. Per-launch and per-sync fixed cost

**Question.** A tax paid once per kernel, per graph node, or per host/device handshake, independent of the bytes. Bentley's procedure-call rule. The model is N × c_node, plus a synchronous copy (`params_us`) and a park.

**Headroom.** That tax as a fraction of the step, judged against the ruler. The whole GLM decode launch floor is 172 nodes × 0.852 µs = 147 µs = 0.37 % of a ~40 ms step `[derived, harvest §2, from plan.md c_node and commit 8a603703's graph_nodes]`. A fold inside that floor is under the ruler. V2-Lite is the exception only because its step is small: the router_topk tail is 218 µs, 4 % of that step `[stated, v41-b5-plan-report.md §8]`.

**Instruments.**

| What | Where |
|---|---|
| Gap between kernel sum and window wall | `nsys-gpu.sh` (the header states this difference is the launch gap) |
| Empty-node price | `cnode_probe --time`. GB10: 0.553 µs/node `[measured, tierengine.md §6.2, direct run, clock stated]`. A6000: 0.852 µs `[stated, v41-placement.md, rig-log plan.md]` |
| Host sync inside the step | `stat step` `params_us`, `parks`, `gap_*_us` |
| Node count | a body's graph-node listing (`slots_graph_nodes` for a captured pass) |
| Sampler or readback paid once per token | a host bench of `Sampler::sample` (axis G when it is three passes over one row). Greedy argmax stays on the card (`head.rs`) |

A fresh `Vec` on every logits readback (`Head::logits_to_host` → `to_host_vec`, `head.rs`) is a fixed cost only on the callers that still use it. The serve path does not: `crates/app/src/lib.rs` calls `logits_into` and `slots_logits_into`, and `GpuModel::slots_logits_into` reuses one staging buffer. `GpuModel::logits` and the gate bins still allocate. Harvest N2's line numbers (`model/slots.rs:796`, `model.rs:1507`) are stale; the hot serve call is the reuse path. The 0–7.6 ms Spark band `[stated, harvest §3d]` does not apply to a sampled serve token.

## Hot paths

| Path | What the wall is | The record that names it |
|---|---|---|
| Decode | One generated token, or one verify pass, after the prompt | `time step`, `time pass`, `stat step` |
| Prompt | The prefill that sets TTFT | `time prompt`, `stat prefill split` / `stat prefill lb` (V4.1), `stat prompt split` / `stat prompt lb` (Qwen ubatch, GLM batch) |
| Tier / NVMe | Paged expert misses and fills on a plan that holds experts in an arena | `nvtier`, `tier census`, `nvread`, `residency pass` |
| Spark | One GB10 pool. Host DRAM and card DRAM are the same resource. PCIe and NVMe are 0 for a model that fits | `sparkdec.md` timeline; the same `time step` / `nsys` pair, on an integrated device |

A change that helps one path can be 0 on another. The row prefetch is the example: it pays on GB10 and loses on the A6000.

## Coverage

Closed means a measurement or a derivation already says the axis does not pay, or the mechanism has landed. Owned means an open item another role has. Thin means the idea exists and nobody's current NOW file is on it. Empty means no priced idea.

|  | A bandwidth | B overlap | C in flight | D more at once | E less work | F placement | G fixed cost |
|---|---|---|---|---|---|---|---|
| Decode | Thin: GLM q8_0 gemv. A6000 `_sel` is at the roof `[stated]` | Owned: tier S3/S5. Dual-stream stagger is a lead decision | Closed inside the gemv (`sparkdec` §5.3; A6000 has more issue slack) | Thin: one verify pass exists; the inner read-once check is open | Owned: IQ2_K, KV-by-depth (quant). Shipped MTP head already cuts rows | Landed residency. warmprof W2–W5 still off main | Closed on large MoE (launch floor 0.37 %). Open and small on Spark (router regrid) |
| Prompt | Owned: `q35hybpp` (common), the 12 GB-card walk | Landed for V4.1 (wait 17.8 → 0.8 ms) | Landed: two V4.1 batches in flight | A placed Qwen3.5/3.6 prompt walks units of up to `PLACED_ROWS` (512), each reading a host expert once. Wide arm already reads weights once for `m > 8` | `STREAM_MIN_P` model says 0 (≤ 0.13 s) | Prompt streaming landed. Cross-start residency has no priced gain | MiMo's extra launches are ~40 µs `[derived]`, under the ruler |
| Tier / NVMe | Owned: S2, S11 (unitier) | Owned: S3, S5 | Owned: S2 queue depth, S11 shares | S2's part-sized units (unitier) | S1 landed. S8 (drop the mapping path) is unitier's plan | Owned: S4, S6, S7, S12, S13 | S2's hand-off 0.05–0.15 ms/job is the term that plan watches |
| Spark | Prefetch landed. Residual against 235.6 GB/s is not re-priced on the landed commit | PDL measured slower. Closed | Little's-law form model says 0 | m ≤ 8 stays bandwidth-bound `[stated, harvest]`, so more columns do not leave the roof | Shipped 76k head vs full head is priced below and not confirmed on a Spark | Row locality landed. S7 pinned arena is a contributor's | Router regrid open, −0.6…−1.9 %. Logits `Vec` is not the serve path |

The empty-looking cells are empty on purpose, or they are owned. The ones still open to pricing are in the next section.

## Inventory, by axis

Status words are the inventory's. "Owned" names who queues it. An enabler that moves no time is marked "—".

| # | Idea | Axis | Why |
|---|---|---|---|
| 1 | Per-warp L2 row prefetch, integrated | F (symptom is A) | Row locality, not more bytes in flight. Landed, −1.55 ms/step `[measured]` |
| 2 | Same prefetch on every card | F | Measured loss on the A6000. Closed |
| 3 | PDL across the step | B | Overlap a kernel tail with the next kernel's loads. Measured slower. Closed |
| 4 | Router rescan rewrite | G | Per-launch tail. Pays at 512 experts `[measured]`. Landed |
| 5 | `qwen35moe_router_norm` regrid | G | A second wave on 48 SMs. −0.07…−0.22 ms `[derived, sparkdec §4]`. Open, Spark only |
| 6 | Lookahead-into-next-wave, side-stream L2, `launch_bounds` | C, F | Measured slower. Closed |
| 7 | Second wave, `_sel` tail, more in-flight bytes | C | Model says 0 (`sparkdec` verdict). Closed |
| 8 | Tier S0 records by kind | — | The instrument for B and F. Partly landed |
| 9 | S1 refill | E, F | Stop `give_back` from re-reading pages just released. Landed, +7.07 % ± 1.61 % `[measured, docs/cards/refill-decode.card]` |
| 10 | S2 one reader, part units | C, A | Queue depth so the drive reaches its rate. Owned by unitier |
| 11 | S3 in-layer overlap | B | Hits compute while the miss is in flight. Owned by unitier |
| 12 | S4 off rule | F | When a plan should not page. Partly landed; `GainUnderRuler` deferred. Unitier |
| 13 | S5 gate lookahead | B | Prefetch layer l+1's experts. Premise: LRU-64 recall 67.7 / 73.3 % `[measured]`. Owned by unitier |
| 14 | S6 heat in the slot book | F | Evict by heat, not by tick. Unitier. No time claimed |
| 15 | S7 pinned arena | F | The arena's memory kind on GB10. Contributor. Enables the in-place read |
| 16 | S8 drop mapping path | E | `mapping_reads` and `drops` to 0. Unitier |
| 16 | S9 lift copies | — | One owner. No time claimed |
| 16 | S10 one budget solver | F | One placement of card, KV, arena, cache. Unitier. No time claimed |
| 16 | S12 `Kind::State` | F | Sealed states on the reader. 32K restore 0.18 s `[derived, tierengine S12]`. Unitier |
| 17 | S11 shares, S2b tiles | C, A | More of the drive's queue during a step. −7 ms/token `[stated]` only if S2's fill stays ≥ 0.6 ms. Unitier |
| 18 | S13 N-node placement | F | Where the book sits across Sparks. Contributor holds the placement half |
| 19 | `HOST_LOCK` | F | Page-cache budget. Landed as a lever |
| 20 | `q35hybpp` | D, A | A placed qwen35moe prompt walks units of up to `PLACED_ROWS` (512) on the wide arm (`placed.rs`, `body35.rs` `placed_open`); `--prefill pass` uses passes of at most 8 ids, and `auto` also uses a pass for a prompt or final tail of at most 8 ids. What is left is the host union's compute, and `xstream` for this family. Unmeasured on a 12 GB card (`docs/cards/q35pfa.card`) |
| 21 | `q3kvdepth` | E, A | Fewer KV bytes as depth grows. Owned by quant (a quant format, and a quality call) |
| 22 | `v41lookup` default | E | Positions per pass from a lookup draft. A default change: ask the lead before it is turned on. The gain `[stated]` is ≤ +10 % code, ~0 prose |
| 23 | Chooser state by head class | E | 0…+2.5 % `[derived from an assumption, adaptive-picks §3.1]`. Superseded by the L2 recut |
| 24a | Multi-slot width | D | A round prices its slots' widths together by its rows and wall (`runtime::width::Batch`). Landed |
| 24b | n-gram beside MTP | E | ≤ +11.4 % ceiling `[derived, adaptive-picks §3.6]`. Ledger only |
| 24c | Residency carried across starts | F | Gain unknown. Ledger only |
| 24d | Held-out residency constants | F | Gain unknown. Ledger only |
| 24e | Ubatch arena by prompt length | D, F | Arena sized to observed prompts. Gain unknown. Ledger only |
| 24f | Lookup confidence by match length | E | Reports p = 1 for every match. No separate gain past 22 |
| 25 | 3.11–3.17 | — | Model says 0. See the split under this table |
| 26 | headpick L0 one owner | — | Enabler. Landed |
| 27 | headpick pick + second list | E | Superseded 2026-10-11. Closed |
| 28 | headpick L2 warm spare rows | F, E | +0.6…+2.9 % en/code `[stated, headpick §0.1]`. Design only |
| 29 | Neutral-base warm head | F | Simulated, rejected |
| 30 | warmprof W1 | F | Per-stream heat. Landed, bit-identical at N = 1 |
| 31 | warmprof W2/W3 | F, E | Returning a conversation does not re-fill. Branches, not on main. Up to 5.2–5.7 s on the paged tier `[stated]` |
| 32 | warmprof W4–W7 | F | Same saving, not started |
| 33 | DSpark / MTP verify | E, D | Acceptance cuts target passes only when the verify reads weights once (D). The session issues one pass (`app/src/mtp.rs` `pass_slots` → `verify_slots`). Field 1.93× is a third-party figure `[stated, quant-decode §Q3]` |
| 34 | Expert bucket, grouped GEMM | D | The structure Q3 says a k-token verify needs. Tile path landed |
| 35 | Expert re-quant IQ2_K / IQ3_K | E | Fewer weight bytes. Use −7.4 ms/token (~−19 %) `[stated, approved]`, not the −31 % byte figure, as the gain. Owned by quant |
| 36 | Dense re-quant IQ4_K | E | −2.3 ms/token `[stated, estimate]`. Quant |
| 37 | QTIP / EXL3 / AQLM / QuIP# / VPTQ / HIGGS | E | Conditional, GPU-resident experts only. Quant. Not opened |
| 38 | W4A16 / AWQ / W8A8 / W4A4 / MXFP4-as-activation | E | Rejected: bytes do not fall, or M = 1 is already on the bandwidth roof |
| 39 | MoE-Infinity / ProMoE | B | Rejected for DRAM. The NVMe form is S5 |
| 40 | Q2_0 × Q8_0 AVX2 dot | A | Host kernel toward the host roof. Landed |
| 41 | Fused sampler, one threshold | G, D | Three filters, one pass. Measured negative (`docs/contrib/sampler-threshold.md`). Closed |
| 42 | Spark logits into a fresh `Vec` | G | Serve already reuses the buffer (this file, axis G). Closed for the serve path |
| 43 | sm_121a block-scaled FP4 MMA | A | Pays only if a MiMo MXFP4 prefill is compute-bound against today's IMMA. Decode and m ≤ 10 predict 0 `[derived, harvest §5]`. Tracker |
| 44 | NVFP4 storage | E | Same bytes as Q4_K. Rejected |
| 45 | Reduced Qwen3.8 MTP head | E | The shipped list is already cjk-76000 (`head_list.rs`). The full-vs-shipped byte saving is derived below. Spark has not confirmed it |
| 46 | GLM q8_0 m = 1 gemv | A | +13 % if the card term is that kernel `[stated, harvest P0.2, plan-triage.md:2055, not read]`. The ledger page was not opened |
| 47a | Relaxed acceptance | E | Lossy. A lead decision. Ledger |
| 47b | Mamba2 state cache | E | Replay state instead of one lane per row. Ledger, held |
| 47c | Tree drafts | E | Rejected in the ledger |
| 47d | SAM draft | E | Held |
| 47e | Fused argmax | G | 0.22 % `[stated]`. Under the ruler |
| 47f | KV fp8 | E | Rejected in the ledger (lossy) |
| 48a | V2-Lite residual ping-pong | G, E | 45.7 µs/step, ~0.9 % `[stated, v41-b5 §8]`. Under the ruler on that step's own ruler; V2-Lite is not a 0.3.3 hot path |
| 48b | V2-Lite multi-block argmax | G | 42.6 µs, ~0.7 % `[stated]`. Same |
| 48c | V2-Lite router_topk fused into gemv | D, G | 218 µs, 4 % of the V2-Lite step, −0.15…−0.2 ms `[stated]`. The one V2-Lite row above the ruler. Not a current hot path |
| 48d | V2-Lite norm_quant fusion | D, G | 216 µs `[stated]`. Same class as 48c |
| 48e | V2-Lite router kept bf16 | E, A | Load-time f32 decode doubles the router bytes `[stated]`. −0.27 ms/layer-sum `[stated, v41-b5 K5]` |
| 49 | mistral.rs copy list | — | Architecture notes (one GGUF/HF boundary, kernel pairs, CPU KV f16). No single term |
| 50 | Multi-Spark tensor parallel vs a split | F, A, C | Placement across pools, and small-message latency on the link. Both are in the tree: remote expert tiers, and tensor parallel for GLM-5.3-Flash with its step across machines not built (`dgx-spark.md` §9.4) |
| 51 | LES repack; per-layer mixed precision | F, E | Contiguous batches are F (and then A, if the rate rises). Mixed precision is E. Unitier owns LES. The iPad 2.97 → 9.81 tok/s figure is their claim `[stated]`, not a bloomery measurement |

**3.11–3.17 (inventory 25), each already model-0 in `adaptive-picks.md`.**

| # | Pick | Axis it would have been | Why it is 0 |
|---|---|---|---|
| 3.11 | `NOMINAL_REPLY` as a running mean | F | An error in the constant misroutes only a second-order band of prefixes |
| 3.12 | `STREAM_MIN_P` priced by xsplit | D, A | ≤ 0.13 s of TTFT `[derived, adaptive-picks]`. Worth a code lift, not a speed round |
| 3.13 | `GEMM_FROM` | D | The threshold already sits on the byte crossover |
| 3.14 | K = 512 half-idle lanes | G | sparkdec lever 1d, ≤ −0.09 ms on GB10, 0 on the A6000 |
| 3.15 | Sampled requests draft differently | E | The per-slot chooser already narrows |
| 3.16 | 16 NVMe fill threads | C | A drive fact, at the knee by the code's own note |
| 3.17 | `HOST_LANES`, `LANE_PREFETCH` | C, B | Machine facts. `glmprefetch-ab.card` predicts 0…+5.7 % and has no result line |

**Harvest dispositions that are not their own inventory row.** §3a (already held: m-column GEMV, split-KV, KDA/GDN, fused router+quantize, moe align, QK-norm+RoPE, graph with device words, n-gram draft) is D or G and already in the tree. §3c (CUTLASS int4 tricks, Hopper/Blackwell-only kernels, NVFP4 GEMV, split-K LoRA) is A or E with a model-0 reason: the `_sel` kernels are on the byte roof, or the instruction has no consumer in our files. They are not re-opened.

**`common/followups.md` performance lines**, not in the 51. They are common's list. Classified so they are not queued twice.

| Item | Axis | Price |
|---|---|---|
| The ring pass's `RouteBlock` (`ring_gemm.rs`, Qwen3.5's placed `ring_pass`) reruns SwiGLU over all slots, up to 3 extra passes a layer | E, D | A `_sel` form would do it once. Common, before their timed A/B. No derived milliseconds in the note |
| MiMo `_8_cols` handoff, 47 launches a two-row pass | G | 47 × 0.852 µs = 40 µs `[derived, A6000 c_node]`. Under the ruler on a prompt |
| A column router, 47 further launches | G | Same 40 µs |
| Flash K192 variant folds 3 per-slot launches | G, D | Three × c_node. Under the ruler |
| Flash partials sized to the pass's rows, ~5 MB scratch | E | Scratch, not a step term, unless a load is refused for it |
| Pitch-aware remap retires n+1 maps | E | Copies removed. A kernel change. Common |

## Gaps still to price

These are the cells the matrix marks thin, with a derivation where one exists. None of them is a default, a public API, or a gate change. Those wait for the lead.

### 1. Decode D: routed experts on a verify are still one column each

Read at `d90151ed`. A drafted round is one `verify_slots` pass (`crates/app/src/mtp.rs` `pass_slots`), and `run_slots` enqueues it once. For a Qwen pass of `m ≤ 8` rows, `qwen3moe::slot_pass` calls `dispatch::ffn` once with that `m` (`slot_pass.rs`, `total`).

Two kernels, two answers:

- For projections that call `site::kgemv`, Q4_K uses `enqueue_gemv_q4k` and shares each weight row across all `m` columns. Q5_K uses `enqueue_gemv_q5k_sel` with one zero-id slot per column; each slot reads the row separately, so axis D remains open. Dense FFNs also have per-slot paths: fused Q4_K gate/up and Q4_K/Q6_K down (`plan.rs`, `dispatch.rs`). Reuse depends on the actual entry.
- Routed gate, up and down at `m ≤ 8` call `enqueue_gate_up` / `enqueue_gemv_q4k_sel` with `n_slots = m * slots` (`dispatch.rs`). `q4k_gemv_sel`'s body is the m = 1 core, `col0 = slot` (`q4k_sel.rs`). Slot `s` reads expert `sel[s]` on its own. Two rows that pick the same expert read it twice. `q4k_gemv_tiles` already reads one expert's row once for up to eight of its slots, and this arm does not call it. The wide arm (`m > 8`) is a different path and does read each weight once (`wide.rs`).

The 2026-09-19 1.69× is not the prediction. The open term is only the routed overlap. Harvest's uniform-routing union for Qwen3.8 is u(5) = 4.80 against 5 row-reads, and the routed term R is 1.52 GB a token `[stated, harvest §2, from dgx-spark.md §9.7]`:

```text
(5 − 4.80) × 1.52 GB = 0.304 GB a 5-row window
at 235 GB/s: 1.29 ms     at 200 GB/s: 1.52 ms     [derived]
```

Under independent uniform routing, 1.3–1.5 ms is an expected routed-read saving, not a guaranteed floor: five disjoint routed sets save zero repeated reads. Routing with greater overlap saves more; overlap was not measured here. The 200–235 GB/s band is the harvest's pool assumption, not a new measurement. On an assumed Spark window of a few tens of milliseconds, the uniform estimate exceeds a 1 % ruler; the window was not re-measured. The host-tier leg of an A6000 verify was not traced, so this estimate applies only to the whole-model Spark split. Any guaranteed shared-expert reuse is a separate term.

An experiment on branch `perflab`, not on main, pointed the m ≤ 8 joined routed arm at that tile entry for more than one row, and only when gate and up are the same Q4_K or Q5_K and the down is Q4_K or Q5_K. One row stayed on `_sel`. It added no lever and no default change. The token bar was the verify gate against the plain steps.

The ws A/B (card `pf1-tile-m5`, A6000, Qwen3.6-35B-A3B Q4_K_M, m=5, `--prefill pass`) passed that token bar and missed the time bar. `gate_qwen35moe_verify` passed; the captured counts were decode 520, a 5-row pass 617, and an 8-row pass 629. The scored row is `time prompt` after load and after the prefill-graph capture. Median `_sel` minus tile was −0.079 ms (tile slower). The four pairs, `_sel` minus tile, were −0.109, −0.079, −0.077, and −0.080 ms. The tile arm was branch `perflab` at `981a6cec`; the `_sel` arm was a second release build of the same tree with the two `m > 1 && ffn_tiles` branches in `dispatch.rs` forced off. Both arms ran under this card's lease. The lease logs are not in this repository. A second sitting on the same card, at `dac9a5e1` (docs-only after `981a6cec`), gave a median of −0.102 ms; its four pairs were −0.098, −0.098, −0.109, and −0.106 ms. The band was 0.15..0.35 ms saved. Both sittings are under it. The edit is parked, with no switch and no PR.

### 2. Spark E: full MTP head vs the shipped 76k list

The unset lever is already the shipped list when the tokenizer matches (`crates/model/src/arch/qwen35moe/head_list.rs`, `SHIPPED` / cjk-76000). The saving against the full head, from the constants in `docs/research/headpick.md` appendix A (`VOCAB` 248320, `ROW` 2720 B, 3 walks, the same arithmetic that prints `T_C` 35.424 ms):

```text
bytes/window = 3 × (248320 − 76000) × 2720 = 1,406,131,200 B = 1.406 GB
A6000 at 701 GB/s: 2.006 ms/window     (37.43 − 35.424 ms, matches the appendix)
Spark at 200 GB/s: 7.03 ms/window      [derived]
Spark at 235 GB/s: 5.98 ms/window      [derived]
```

The 200–235 GB/s band is the harvest's assumption for the GB10 pool (`dgx-spark.md` §5, via harvest §2), not a new measurement. The harvest's 5.0–8.1 ms/window is their scaling of an A6000 ledger band; the byte arithmetic above is the figure to use for this list. This is a landed default, not a new cut. Confirming it is a same-binary `MTP_HEAD_ROWS=full` against unset, and it needs a Spark. The ws pool has no Spark. Changing the default is a lead decision and is not proposed: the default is already the list.

### 3. Decode B: two independent sequences, staggered by a layer

`docs/roofline.md` derives a throughput factor of 1.67× if the host leg (22.6 ms at the old 147.7 GB/s) and the card leg (33.9 ms) overlap across two independent sequences, so the wall is the max rather than the sum. The same paragraph records the two conditions: the sequences are independent, and the factor is throughput. A single stream cannot overlap layer n with layer n+1. Two requests today run at 0.62–0.68× each one's solo speed `[measured, docs/performance.md]`, which is the contention figure, not this schedule. Opening a serve schedule is a design decision. It is listed so the cell is not mistaken for an empty experiment. It is not queued for a run.

### 4. Decode A: GLM q8_0 m = 1 gemv

The decode kernel is `q8_0_gemv` (`crates/gpu/src/q8f32.rs`): one warp owns one output row, one column. The launcher `enqueue_q8_0_gemv` sends `m = 1` there and `m > 1` to `q8_0_gemv_mcol`. A GLM decode step is the point `(1, 1, Step)`. The m=1 sites are `body::gemv` (it refuses a weight that is not Q8_0), `Weights::q8_gemv` for a Q8_0 down, `select` on `indexer.attn_q_b` with the step's `m = 1`, and `Head::enqueue` when `output.weight` is Q8_0. File bytes and device bytes are the same: 32 values are 32 code bytes plus one f16 scale (`q8f32.rs` layout), 34/32 bytes a value.

The trunk step (layers 0–44; layer 45 is the unused next-token layer) reads these weights. Widths are the pinned header census (hidden 4096, KDA k = v = 64, d = 128, latent q 1536, kv 512, heads 64, absorbed qk/v 256, indexer 32 × 128, dense ff 12288, shared ff 2048, vocab 154880) and `Hparams::kda_dims` in ggml order `[k, n_rows]`. The joined q·k·v and the joined latent stack are one launch each. The header census matches that typing: attn q8_0 405 = 34 × 9 KDA matrices + 11 × 9 latent matrices, and attn f32 315 = the remaining vectors.

```text
serial, on the wall (mixer front, dense down, head)
  KDA ×34   qkv 24576×4096  106,954,752
            f_a 128×4096        557,056
            g_a 128×4096        557,056
            beta 64×4096        278,528
            f_b 8192×128      1,114,112
            g_b 8192×128      1,114,112
            out 4096×8192    35,651,584
            one layer       146,227,200    ×34 = 4,971,724,800
  latent ×11  stack 2304×4096   10,027,008
              q_b 16384×1536    26,738,688
              out 4096×16384    71,303,168
              one layer        108,068,864    ×11 = 1,188,757,504
  dense down ×3   4096×12288    53,477,376     ×3 =   160,432,128
  head ×1         154880×4096                     674,037,760
  serial sum                                          6,994,952,192 B
```

At the A6000 large-gemv plain read, 567–701 GB/s (`docs/HARDWARE.md`), that sum is 12.34 ms at 567 and 9.98 ms at 701 `[derived]`. Those are byte times, not a measured kernel rate. Launches at or above 26.7 MB are 6,761,545,728 B, 96.7 % of the sum (head, joined qkv, latent out, dense down, KDA out, latent q_b). The rest of the serial sum is the low-rank KDA projections and the 10.0 MB latent stack.

Same kernel, off that sum: `indexer.attn_q_b` is 11 × 6,684,672 = 73,531,392 B on the branch stream beside `attn_q_b`. The shared-expert down is 42 × 8,912,896 = 374,341,632 B in the shadow, concurrent with the host leg (`program.rs`). Other kernels, named so they stay out of this count: `q8_0_gemv_heads` is `attn_k_b` and `attn_v_b` (8,912,896 B each a latent layer); `q8_0_gemv_mcol` is the prompt front; `hc_pre_q8_0` is the stream mix; `ds41_shexp_gate_up` is gate and up (each the same size as that block's down, which is why `glmgemm2-pp.card` calls a dense layer's three matrices 160.4 MB); card experts are K-quant `_sel`.

`glmgemm2-pp.card`'s 26.7 MB and 71.3 MB are these q_b and out weights. `glmswap-pp.card`'s ~195 GB/s is `q8_0_gemv_mcol` at m = 8 on a prompt front, where each weight is reread per chunk. It is not this launch.

The front gemvs, the three dense downs, and the head are on the wall: the mixer and the dense block run before the host go, and the head runs after the chain. The shared-expert down is the launch under the host bridge. The cards that price a whole step put the host leg ahead of the card front. `glmslots-ab.card` derives a card front of 12.8–15.6 ms beside a host leg of 29.6–31.4 ms. `glm-public.card` derives a host-experts-only step of 53.6–61.6 ms from a per-layer card front of 0.30–0.41 ms plus the host leg, and says nothing of GLM has been timed on the A6000 (the box reading it cites is a 3090 feed at 53.7 ms a step). Neither card is this kernel's GB/s.

V4.1 B9 (`docs/v41-placement.md`) measured this kernel on another model: launches of tens of MB at 567–701 GB/s, with the miss in the small launches. That is the nearest achieved rate, and it puts the byte-dominant launches on the plain read. It is not a GLM m = 1 trace.

Confirm: not started. The pass needs the kernel on the wall, the bytes above, and an existing achieved GB/s more than ~1 % under 567–701. The first two hold. No card or nsys table has the third for `q8_0_gemv` at one column on this model, so the ruler clause fails. No nsys. Harvest P0.2's +13 % and its 11.6–13.4 ms inside a 41–43 ms wall stay `[stated, plan-triage.md:2055 and harvest §2, pages not read]`. Quant still owns quant formats. No lever, no default change.

### 5. Spark G: router regrid

`sparkrouter`, −0.07…−0.22 ms on an 11.50 ms step `[derived, sparkdec §4]`. The center (−1.3 %) can clear a six-round ruler (±0.8 %) and can miss a four-round one (±1.0 %). Spark only. Gap 4's confirm did not start, and this is not a ws job.

### 6. Decode F: warmprof W2–W5

The largest unmerged placement saving in the inventory: up to 5.2–5.7 s of re-fill on a returning paged conversation `[stated, warmprof]`. W1 is on main. W2/W3 are branches (`warmw2`, `warmw3`, `warmw3g`, `warmw3s`), not owned by the current unitier / quant / common NOW files. Rebasing those branches onto `origin/main` is a coordination step, not a new idea. No rebase while a lock on those tracks is live.

## What this file does not do

It does not change a default, a lever's unset value, a public API, or a gate's pass band. It does not queue S2, S3, S5, LES, IQ2_K, IQ3_K, IQ4_K, or `q35hybpp`. It does not treat a harvest figure whose source page was not opened as a measurement. Gap 1's edit is the parked experiment above.
