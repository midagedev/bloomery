# tierengine — one tier engine for expert placement across card, RAM and NVMe (design, Fable 5.1, 2026-10-11)

This is the design of one tier engine for MoE experts held on the card, in host RAM and on NVMe. One ledger, one NVMe
reader and one memory policy replace today's two readers, two heat sources and two memory policies (§1). The same code
serves a discrete card and a DGX Spark (GB10), whose CPU and GPU share one memory pool, so the card reads host pages
in place.

S1 (refill) and part of S0 had landed by main `9bd7d577`, and S7 (the arena in pinned memory) and its card leg ((h)
item 1) have landed since; a contributor holds S13's placement half (§5). DGX Spark measurements answered five of the six questions
(k) asked and changed three inputs: a hit's price on GB10 now rests on whole-line loads over 128-B aligned parts, a fill
at QD 3 (three reads in flight to the drive) takes longer than (j) interpolated, and one link rail moves faster than the
cited rate (§6).

§1 and §2 describe the engine the design started from, §3 is the target, §4 the cost model on the maintainers' box,
§5 the steps and their status, §6 the DGX Spark measurements, §7 the work in flight when the design was written, §8
the decisions and §9 the prior work. The provenance, tags and terms follow.

Round `tierengine` (unitier, read-only; no code, no box run, no lease). Tree read: the r4d branch at
WIP `7cec21ad` (R4d-a landed; r4db2's uncommitted edits in `lane.rs`, `record.rs`, `gate_qwen38_serve.rs` not read).
Round `tierengine2` (Fable 5.1, 2026-10-11) added §3 (g)–(l) (host RAM, unified memory, the GPU's NVMe path, read
granularity, N-node Spark, KV + placement), S10–S13 in §5, six rows in §7 and questions 7–12 in §8; the Spark numbers
are the contributor's (#13, #17), the main-tree lines (`workstation.rs`, `ctx.rs`, `checkpoint/`) are read on main.
Tags: [measured, source] a printed number; [derived] arithmetic from code and measured terms; [assumption] stated.
Lines are the r4d tree's unless a path says otherwise. "S" = slots a paged layer carves, "B" = union batch (16),
"fill" = one expert's three parts read into its slot (Qwen3.8: 3.134 MB, slot 3.16 MB).
A "part" is one of an expert's three tensors (gate, up, down); "WC" is write-combined host memory
(`CU_MEMHOSTALLOC_WRITECOMBINED`); "QD" is queue depth, the reads in flight to the drive. Design notes cited by path
or round name (`r4/decisions.md`, `plan-triage.md`, rig-log, spark2x, attach-design, the r4d rounds) are not in this
repository; they name a source.

## 1. The engine as built

This section is a snapshot of the r4d tree at WIP `7cec21ad`, before S1 and R4d-c landed; its line numbers are that
tree's and were not read again.
Two syscall-level readers of expert bytes exist: (A) `O_DIRECT` `pread` into the anonymous arena — one body,
`read_pick` (`crates/gpu/src/host/nvtier.rs:2240`: `give_back` of the evicted slot's pages at `:2243`, then three
`read_into` at `:2250` → `DirectFile::read_span` `crates/engram/src/direct.rs:111`, `read_at` `:205`, `O_DIRECT` `:38`) —
driven by three callers on three executors; (B) page faults on the GGUF mapping. The lead's four gaps hold; the
corrections are in the last column.

| # | Path | Who / when | Executor and chain | Locks | Verified against the lead's reading |
|---|---|---|---|---|---|
| 1 | decode miss fill | step port `experts_into_on` `crates/model/src/moe.rs:2386` → `ensure_pinned(Claim::Step)` `:2407` (union step `:2574`) → `fill_missing` `nvtier.rs:2108` → `fill` `:2150` → `read_pick` | `threads::pool().for_each_chunk` `nvtier.rs:2153`: the chain's own compute pool, the card's handoff page waits (`step.rs:1039`, "the page holds the handoff until the signal") | books `Mutex` at claim `:2083` and publish `:2130`; none during the read; the pool's dispatch mutex | gap 2 confirmed: synchronous, QD ≈ misses in the layer (≈ 1) |
| 2 | prompt batch fill | `serve_batched` `moe.rs:1720` → `prefetch_pinned` `:1768` (batch g+1 queued while g computes, `:1780`) → `prefetch` `nvtier.rs:2171` → `Inflight::queue` `:2344` → `Reader::submit` `nvtier/reader.rs:363` at `Kind::Prompt` (rank 0, `:51`) → `Ticket::wait` `moe.rs:2068` → `settle` `:2383` | `nvtier-fill-0..15` (`NVTIER_FILL_THREADS` `nvtier.rs:105`, spawned `reader.rs:329`, Float placement) | books (claim, publish); queue `Mutex` `reader.rs:210`/`:367`; open-jobs `:217`; per-job latch `:63-64`; `notify_all` `:393` | the one ranked queue; ensure-ahead one batch deep |
| 3 | victim fill (demotion) | lane job `serve` `lane.rs:604` under the receive lock `:682` → `prepare_victim` `:618` → `FileSwap::prepare_victim` `swap_source.rs:843` → `ensure_victim` `nvtier.rs:1939` → `fill_victims` `:2431` submits `Kind::Victim` (rank 1, `reader.rs:52`) and waits → publish; also on the caller at `swap.rs:4159` (boundary/reset/load, `:1891`) | the reader's threads; the lane thread blocks under its receive lock (`swap-staging-0..3`, `LANE_THREADS` `lane.rs:48`, spawn `:878`) | books; the reader's; the lane receive lock held through the wait | measured 2.41 ms a job [r4da]; a non-arena id still faults the mapping on the lane thread (`populate_run`, `host_lock.rs:373`) |
| 4 | promotion stage (B) | lane `stage` `lane.rs:554` → `source()` `swap_source.rs:720`: the slot when `open_read` `:793` found it filled, else the mapping `.data` `:767` (faults on the lane thread) → ring (`MappedHost`, `lane.rs:296`) → H2D on the copy stream `swap.rs:2116` → landing, `late` at `:2434` | lane thread, copy stream | books (`open_read` `nvtier.rs:1612`), receive lock until the stage (`:657`) | after excl the arena id reads its slot; the mapping path remains for an unfilled id |
| 5 | xstream fill (B) | `xstream.rs:521` `open_read` → `:526` `source()` | `xstream-fill-N` (`xstream.rs:1648`) | books | a fourth consumer of `source()`; mapping faults for host-segment ids |
| 6 | host segment (B) | `tier_or_mapping` `moe.rs:2294`: host-segment ids compute from the mapping | the pool, page faults inside compute | none | populated, not locked (`BLOOMERY_HOST_POPULATE` 1, `HOST_LOCK` 0) |
| 7 | dropper | `end_read` `nvtier.rs:1667` queues `:1781`, `notify_one` `:1804`; `drop_layer` `:1757` on the caller; `PageDrop::release` `host_lock.rs:690` | `nvtier-drop` (`nvtier.rs:1394`, loop `:1068`) | `DropShared` `Mutex` `:1043` | gap 3: a second memory policy beside the arena's `give_back` |
| 8 | books | `LayerArena` `nvtier.rs:614` (`of_id`, `reading` atomics `:641`), `SlotBook` `:852` (state `FREE/FILLING/FILLED` `:198-200`, tick `:859`), `Books` `:1151` under `books: Mutex` `:1102`; `claim` `:2071`, `claim_with` `:369`, `claim_slots_pinned` `:359`, `pick_slot` (LRU, transient first) `:281`, `publish` `:497`, `publish_pinned` `:520`, `release` `:1693` (give-back, lock `:1708`), `pin` `:1739`; `Claim::{Step,Keep,Transient}` `moe.rs:2011` | — | one `Mutex` per tier, per `ensure` | the id ledger is the tier's (design Q1) — but it holds no heat |
| 9 | heat | `SwapRule.counts` `crates/runtime/src/swaprule.rs:458`, `observe` `:845`, bump `:918`, decay `:957`, fed by `Tally` `swap.rs:370` (`step.rs:1004`, `:1123`) → `end_pass` `swap.rs:2278` → `fold` `:2375`; the arena's LRU is the slot tick; `NvTier::Heat` `:1010` / `slot_heat` `:1470` have no production caller | step thread | — | two heat sources, two movers (design Q2's "one count source" is not built) |
| 10 | memory | `AnonMap::new` `direct.rs:149` (`map_anon`), one per tier `nvtier.rs:1364`; `give_back` `nvtier.rs:190` (the one door) → `dontneed` `direct.rs:168`; before every refill (`:2243`) and on release (`:1693`); `MappedHost` (`cuMemHostAlloc` `graph.rs:453-456`) for rings and words only; no `integrated` branch under `host/` (`raw_device_integrated` `crates/gpu/src/lib.rs:1778`, `workstation.rs:46,207`) | — | — | gap 3 confirmed: refault per fill, page cache unbudgeted, the integrated arena a separate plan |
| 11 | counters | `NvTierStats` `nvtier.rs:876`, `Stats` `:933`; `count_fill` `:2268` keys `union_fills` on `pinned`; `pinned_claim` `:961`; `KindStats` `reader.rs:136` (`jobs`, `units`, `waits`, `wait_ns`; no `run_ns`); record `nvtier` keys `crates/gpu-gates/src/record.rs:1640-1666`; `TierCensus` `host/census.rs` | — | — | gap 4 confirmed; the timing runner keeps no arm stderr [r4/decisions.md:81], so no card's "read beside" field is readable |
| 12 | placement, levers | `expert_nvme_tier` `placement.rs:2882` (callers `qwen38_cards.rs:339`, `qwen35moe/place.rs:507`; V4.1/GLM on r7 only); `nvme_arena_of` `:2838`, the H/2 rule `:2864`; `ArenaFloor` `:2644`, union term 2B+1 = 33 `:2686`; `HostNeed.arena` `workstation.rs:567,601,628`; `room_for` `levers/src/lib.rs:1338`, `residency_at_plan` `:1442`, `residency38_at_plan` `:1669`, `glm_residency_at_plan` `:2226`, `PlanTier` `:1235`, `ResidencyWhy::Paged` `:1089`, the "cold" why `:1273` | plan time | — | the off rule is keyed on a plan shape, not derived (design Q6) |

Threads on a paged Qwen3.8 load: the pool (32), the step thread, `nvtier-fill` ×16, `nvtier-drop` ×1,
`swap-staging` ×4, `xstream-fill` ×n. Schedulers: the pool's dispatch (the chain) and the reader's rank queue; the lane
is a third executor with its own ring and deadline that no longer reads NVMe for arena ids (R4d-a) but still blocks on
the reader under its receive lock, and still faults the mapping for an unfilled id.

## 2. Owners per model

Each row is one owner that every model shares: the model columns say where each model attaches to it, and the last
column is what an architecture fact keeps per model.

| Owner (one place) | V4.1 | GLM-5.3-Flash | Qwen3.8 | MiMo (`gpu-mimo2`) | What an architecture fact forces (stays per model) |
|---|---|---|---|---|---|
| placement sizing: `expert_nvme_tier`, `ArenaFloor`, `HostNeed.arena`, the dial | r7 A1–A2: `PlanInputs.room` + the call [attach-design 1.1]; no `row_table_tier` (engram by the role rule) | r7g: two calls, trunk at `room − draft.host.expert_bytes`, NextN `plan_nextn` [attach-design Part 2]; PR #16 `pool_card_bytes` first | `qwen38_cards.rs:339` | `qwen35moe/place.rs:507` (the room read in `describe`) | slot bytes per layer (V4.1 layers 0–1 q5_K down; GLM three part sizes); the draft's own host plan (GLM NextN, Qwen3.8 MTP columns) |
| the slot book (`NvTier` books, claims, LRU) | `attach_tier` through the common `host_run` `model.rs:1710` (A3) | A3 as V4.1; parts 32-B aligned [derived, attach-design] | `body38.rs:1451-1456` (own builder beside the common one) | not attached: `gpu-mimo2/src/body.rs` has no `attach_tier`/`nvme_tier` call (its `host_run` `:780` is a layer range) | nothing: 3-row stacks, file-backed parts, no type check (`nvtier.rs:982-1040`) |
| the I/O scheduler (the reader) | none | none | none | none | nothing — the only per-model input is the part sizes |
| the residency machine + rule (card mover) | `residency_glue.attach_tier` (A4); `residency::at_plan` wired on r7 | `glue.attach_tier(experts.tier())`; `card_experts` 0 on the 3090 → `off` | `body38.rs:1456`; `residency38_at_plan` | unwired | which experts a card kernel can run (`_sel` per quant: no IQ3_S card kernel → GLM 3090 card_experts 0) |
| the step port (`HostLayer`, `ensure_pinned`, the handoff page) | common `TierSlots` seam `moe.rs:2081` (`ensure_as`, `prefetch_pinned` `:2113`); union through `batch.rs:1449-1553` | same | same (+ `wide.paged()` split refusal `wide38.rs:1375`) | same | k, n_used, columns (MTP/NextN widths); V4.1 cuts prompt calls by `call batch`, Qwen3.8 by the width rule |
| seat / serve rules | `serve_seats/ds41.rs` | `serve_seats/glm.rs:637-646` (plan tier) | `serve_seats/qwen38.rs` (paged floor fallback, draft share; R4d-b flips the refusal) | `serve_seats/mimo2.rs` | per-seat checkpoint bytes and the draft's column count; the floor's "widest step" is one statement (host #1a) |
| the records (`plan`, `nvtier`, `residency unset`, census) | A6: `plan` gains paged/arena/S; `generate_ds41` prints `nvtier_of` | as V4.1 (`generate_glm5next`) | `generate_qwen3moe.rs:889` | none | nothing; one `Kind` per fact, `records-refresh` per change |
| the KV pool (sealed states, pinned staging, the cache budget and floor) — §3 (l) | `deepseek41/kv.rs` (window rows, compressed streams, the pinned host shadow) | `glm5next/place.rs:842-915` (latent rows + pool keys, KDA lanes) | `qwen35moe/place.rs:1438` (attention rows, 2 × GDN checkpoints, PLE ring) | `mimo2`: not read | the state's composition only; the pool `checkpoint/{saved,staging}.rs` (`SeqState`, `SaveLane`, `Staging::pinned`) and the budget `placement::ctx::within_cache` + the floor `placement.rs:2455` are common |

Rule for the table: a per-model line that is not in the last column is a copy to lift (today: Qwen3.8's own host-run
build beside `model.rs:1710`, the room field copied per family, and the per-family `*_at_plan` wrappers that map one
`RoomShort` to three `*Why` enums).

## 3. The target architecture

The target makes three decisions: heat moves into the slot book, so one ledger ranks every move ((a)); the reader
becomes the only NVMe reader of paged ids ((b)); and the memory kind becomes a parameter of one memory policy ((d)).
The subsections:
- (a) one slot book per layer across card slots and the RAM arena;
- (b) one I/O scheduler;
- (c) decode prefetch;
- (d) the memory kind behind one policy;
- (e) the off rule;
- (f) records;
- (g) host RAM: its terms, the split, the owner, the page cache's limit;
- (h) unified memory on GB10, and where the arena sits;
- (i) how the GPU reaches NVMe;
- (j) read granularity, queue depth, prefetch depth, bytes in flight;
- (k) N-node Spark;
- (l) KV and placement in one policy.

### (a) One slot book per layer across card slots and the RAM arena

The tier's books stay the id ledger (design Q1, built). What is missing is **heat in the same book**: today the card
mover reads `SwapRule.counts` (decayed per-pass tallies) and the arena reads slot ticks (LRU). Target: `Heat` becomes a
field of the ledger entry — one `u32` count per (layer, id), bumped once per routed pick by the step port's tally
(`step.rs:1004,1123` already walks every id) and decayed per pass by the rule's existing schedule — read by both movers:
the rule ranks card promotions by it (as now, from the same numbers), and `pick_slot` evicts by (count, tick) instead of
tick alone (a transient claim keeps entering cold, R4c's rule). Exclusive tiers hold: `Card` ids hold no slot
(`release` at the landing), `Arena` ids hold one, `File` ids none; the host segment is a fourth, static state. Promotion
and demotion at every boundary are then one comparison on one number: card ← arena when `heat(arena id) > heat(card
victim) + hysteresis`, arena ← file on a miss, arena → file by the coldest count. Derivation: the ledger's read cost is
one atomic hint an id (`LayerArena.of_id`, `nvtier.rs:614-641`, read by `staged_slot` `:1630`), unchanged; the count write is the tally's existing walk (480 picks a token,
one `fetch_add` each ≈ 480 × ~5 ns = 2.4 µs a token [derived]); the rule's `counts: Vec<f64>` (`swaprule.rs:458`)
becomes a view of the book, so no second copy. Proof for the move: a route-trace replay (Mac) must plan the same flips
from the book's counts as from the rule's — a counting identity, no lease.

### (b) One I/O scheduler

**Yes: the reader is the only NVMe reader of paged ids.** The pool's `fill` (`nvtier.rs:2150`) goes; the step's misses
become `Kind::Step` jobs at rank 0 (ahead of Prompt 1 and Victim 2, and a lookahead `Prefetch` at rank 3), the chain
waiting on the job latch as the prompt batch does. The reader's **unit becomes a part, not a pick** (3 units a fill):
at decode a layer's one miss then runs at QD 3 instead of QD 1 (the drive's rate 2.51 → ~5 GB/s, qdsweep
[measured, r4/decisions.md:13]), and a prompt batch's 16 picks become 48 units over 16 threads — the same QD 16 it has.
Budget: the reader's threads are the queue depth (16 = the drive's knee); classes get **shares, not reservations**:
Step and Prompt take any thread; Victim and Prefetch take a thread only while the Step queue is empty (a victim's 2.41 ms
then never sits in front of a miss), with the victim's deadline the lane's window (late = the existing `late` count).
Queue discipline: rank, then FIFO within rank (today's `partition_point`, `reader.rs:383`); a Prefetch job is cancelled
when its layer's Step claim lands first (the slot is then the step's; the prefetch's read is skipped, not wasted).
The pool's synchronous reads disappear: the chain's thread submits and waits, and the hits' compute runs under the
miss's read (see (c), the in-layer term). The lane stops reading NVMe at all: a non-arena victim (middle band) is an
arena-less plan, where the mapping path stays by the dial's own choice.

### (c) Decode prefetch

Three routing signals, by when they exist: (1) the **previous token's ids at the same layer**: a whole step early, but
by construction already resident (S ≥ 33 ≫ 10 picks; the LRU cannot evict them within a token) → predicts no miss,
worth 0 beyond "do not evict"; (2) the **prompt keep set** (R4b, built): warms the arena before the first step and is
why the r4b window shows ≈ 0 fills a token; (3) the **gate lookahead**: layer l+1's router applied to the residual
*before* layer l+1's attention (available when layer l's MoE output lands, one attention sublayer early) — the Mixtral
offloading heuristic, "possible to get an accurate guess of next layer's experts by applying next layer's gating
function to previous layer's hidden states" (https://arxiv.org/html/2312.17238, §3.2; its recall is only a figure
there). It is the only signal that names a *miss* before the chain reaches it. Cost: one router GEMV per layer on the
card (512 × 2048 f32 = 4 MB [assumption on the router's type], 48 layers 200 MB a token ≈ 0.33 ms at 600 GB/s
[derived]) + 48 launches × 0.85 µs = 0.04 ms [derived, c_node]; the predicted ids ride the **same handoff page** as
layer l's ids (no second D2H: 48 extra bridges would cost 48 × 36 µs = 1.7 ms [derived, bridge floor]). Break-even
(design Q8, held): r > k/(2m); at k = m the bar is r > 0.5; expected r 0.6–0.85 [assumption; Mixtral-8x7B's figure is
not numeric in the text].

Resource timeline per token, serial today (the wall is the **sum**):

| Unit | r4b shape, warm window (A6000 plan a, 27G+20G, after P 512, n 96) | gate plan (3090 bytes on the A6000, S 154) |
|---|---|---|
| card SMs | ≈ 26 ms [derived, r4dc §3] | ≈ 26 ms [assumption: same attention, fewer card experts] |
| host CPU (host experts) | 136–238 slots × 23.6 µs = 3.2–5.6 ms [derived] | 480 picks × 23.6 µs = 11.3 ms [measured 3090 replay, design Q6] |
| NVMe in the chain (fills) | ≈ 0–1 fill × 1.05 ms [derived: token 32.3 ≈ floor 31.7; plan-triage.md:158,277] | 45.7 fills × 1.0–1.1 ms = 46–50 ms [measured: 4,385 fills / 96 steps, gate_nvtier untimed; 1.0–1.1 ms a fill, R1 probe] |
| NVMe off the chain (victims) | 5.75 × 3.134 MB = 18 MB, 2.7 ms of drive at QD16 [derived] | same order |
| PCIe (flips) | 5.75 × 3.16 MB / 26.3 GB/s = 0.7 ms, async [derived] | same |
| host DRAM | 0.4–0.75 GB (12–23 GB/s of ~100) [derived] | ~3 GB at ~130 GB/s ≈ 25 ms, inside the host term |
| **wall** | 32.3 ms = card + host tail; fills ≈ 0 | 83–95 ms [measured untimed, r4b2] = 26 + 11 + 48 (sum) |

At the gate plan the SMs idle ~70 % and the host ~85 % of the token while the drive runs at a fifth of its rate: the
step is latency, not demand, so the levers are flow levers, in this order [all derived]: bulk (part units, QD 1 → 3:
fill 1.05 → 0.63–0.75, −14..−19 ms), the refault's removal (refill: 0.35–0.5 ms a fill, −16..−23 ms — the two overlap:
their sum is bounded by 1.05 − 0.46 (the QD 16 rate's floor) = 0.59 ms a fill, −27 ms), async inside the layer (the
hits' ~9 host picks × 23.6 µs = 0.21 ms compute under the miss's read, ≤ 48 × 0.95 × 0.21 = −10 ms, sum order kept by
storing per-expert outputs and summing in list order), then the lookahead across layers at recall r: hides r × the
remaining chain fills, costs (k − r·m) wasted reads on the drive off-chain (at r 0.7: 13.7 extra reads, total drive
186 MB a token = 27 ms at 6.77 GB/s, under the token: fits). Ladder at the gate plan: 85 → 66–71 (bulk) → ≥ 58
(+refill, the rate floor) → 48–58 (+in-layer) → 40–48 (+lookahead r 0.7) → 37 (no fills: card + host). Fraction of the
fill wall that can leave the chain: ≈ 40–56 % without any predictor, r of the remainder with one. At the r4b warm
window every fill lever predicts 0 (fills ≈ 0): an A/B there sits inside the ruler and `lease_take` refuses it — the
cards run at the gate plan (`BLOOMERY_GEN_PLACE=gate`) or at n ≥ 1024.

The one term derivation cannot close: **r at k = m** (prose and chat). The id trace cannot give it (the pre-attention
residual is not in the trace), so the probe is an engine record: a functional run with the lookahead router beside the
real one, counting `lookahead_hit / lookahead_picks` per layer — no lease, ≈ 5 min at the gate plan. Expected 0.6–0.85
[assumption]; ≥ 0.6 → open the round at k = m; 0.5–0.6 → k = 1.4 m needs r > 0.7, refuse; < 0.5 → drop (c)'s predictor,
the three slot-level levers stand. A second open term, for the A6000 public shape: the fills-a-token curve over a
1,024-token decode at the r4b shape (expected to rise from ≈ 0 toward 10–25 as routing leaves the keep set [assumption];
functional, ≈ 3 min) decides whether the A6000 rows need the fill levers at all.

### (d) The memory kind behind one policy

`ArenaMem { Anon(AnonMap), Pinned(MappedHost) }` (Q18, one enum at the build). `nvme_tier`
(`crates/gpu/src/model.rs:1779`) picks by the `BLOOMERY_NVTIER_MEM` row and the device's integrated bit: `auto` is
`Pinned` on an integrated device and `Anon` on a card with its own memory, and the same bit makes a `Pinned` arena
write-combined on an integrated device only (`in_place_flags`, `crates/gpu/src/graph.rs:418`). **`give_back` lives in
exactly one place and runs at exactly one time: the release of a slot to the card (`NvTier::release`,
`crates/gpu/src/host/nvtier.rs:1980`), never before a refill** — the next fill overwrites the slot's windows, O_DIRECT
into mapped pages faults nothing, and the arena's RSS is the plan's bytes anyway (`HostNeed.arena`). On `Pinned` it is a
no-op: the allocation is the load's. A `dontneed` on write-combined pinned pages leaves them unbacked, so the next
fill's read fails with `EFAULT` and a CPU read faults (GB10, and an RTX 5090 with `WRITECOMBINED`); on an RTX 5090
without it the pages are a shared mapping and keep their bytes. No stale alias was seen, so Q18's D1 mutant is required
red on an integrated device only ([#109][i109]). The refill round's lever is this deletion. **Page cache budget:** after
(b) the only mapping readers of a paged id are gone (step, prompt, victim, lane stage after excl); the mapping then
holds the host segment + dense tensors, populated at load, and the page cache's size is the plan's host-segment bytes by
construction. The budget is asserted, not managed: `mapping_reads == 0` for paged ids (r4da measured 0, the first
anchor), `drops == 0` once the dropper retires, and a census line `page_cache_bytes` beside `HostNeed` (from
`/proc/self/smaps_rollup`'s file-backed RSS) printed per span. `BLOOMERY_HOST_LOCK=1` on a paged plan keeps the kernel
from reclaiming the host segment under the arena's writes (attach-design Part 2's lower-bound caveat) — a default the
user decides (§8).

### (e) The off rule

gain(mid over off) = Δh·picks·W + Δmisses·S/R − machinery (design Q6). After R4d-a the machinery is measured small
(late 0 / 552, victim wait off the chain, the dispatch mutex no longer taken by fills, the landing's give-back 0.4–0.6 ms
a token [r4da, r4dc §3]), so the rule becomes derivable at plan time with one assumption, Δh (0.517 on the Qwen3.8 3090
replay; a replay figure per family). `off`, with the named why: (i) `NoCardExperts` — the first term is 0 by
construction (GLM on the 3090); (ii) `HostShort`/`MemShort`/`NoRoom` as today; (iii) new `GainUnderRuler { dh, picks,
w_us, floor_ms }`: Δh × picks × W < 0.01 × the plan's step floor **and** S ≥ the layer's paged ids (Δmisses = 0 — the
arena covers the set, nothing to save). On the r4b shape: 0.517 × 136 × 23.6 µs = 1.66 ms ≫ 0.32 (1 % of 31.7) and
S 141 < 212 → `mid` [derived], the word r4dc1 predicted; on a plan whose arena covers every paged id and Δh·picks·W < 1 %
→ `off` by name. The `Paged` clause goes (r4dc2's 36 sites), replaced by (iii), so an unset word always carries a
number. (iv) stays open: the small-host regime where decode is NVMe-bound and the machinery's drive share competes with
the chain's fills — the small-host pair (design Q10) is its ground truth; until it runs, the clause is report-only. No
run-time auto-off (user, 10-06): the machine prints the three terms per window; it never flips itself.

### (f) Records

Keys by claim kind × event, one `nvtier` record per span (prompt, steps): `{step,keep,transient}_{hits,fills}`,
`{step,prompt,victim,prefetch}_{jobs,units,wait_ns,run_ns}` (the reader's `KindStats` gains `run_ns`, so wake and read
split), `fill_{fault_ns,read_ns}` (the refault term, measured not papered), `prefetch_{issued,hit,wasted,cancelled}`,
`mapping_reads` (pinned 0), `drops` (pinned 0 after the dropper goes), `page_cache_bytes`, `late`, `victim_wait_ns`.
The `pinned` key in `count_fill` goes. **The timing runner keeps every arm's stderr** (`<log>.arm<n>.err`) and the
row carries the arm's `nvtier` and `residency` records, so a card's "read beside: late, fills" is a column, not a
wish. FAIL-first: key `union_fills` on `pinned` again → the kind test red; a runner mutant that drops stderr → the
row's beside column empty → red.

### (g) Host RAM: what lives in it, the split, the owner, the page cache's limit

One budget, `HostNeed` (`crates/placement/src/placement/workstation.rs:553-570`, main), owns the room on both machines.
The reading is `MemAvailable` capped by the cgroup (`host_room`, `:926`); on the box `HOST_USABLE` 270.07 GB −
`OS_OTHER` 6.69 GB = `PLAN_ROOM` 263.38 GB (`:118-126`); on GB10 the pool itself (`MemAvailable` 125.2 GB idle, 111.7 GB
allocatable [measured, #13]) less the cards' share (`HostNeed.cards`, PR #16). The terms, their classes and owners:

| Term | Discrete (box) | GB10 | Class | Owner / rule |
|---|---|---|---|---|
| host segment (`experts`: what the CPU computes) | the paged plan's host picks, populated at load, locked under `HOST_LOCK` (§8 Q4) | 0 — the CPU computes nothing (`docs/contrib/dgx-spark.md` §8: 20 cores beside 48 SMs on one 273 GB/s add contention) | fixed | placement (`expert_nvme_tier`'s H/2 rule, `placement.rs:2864`) |
| `tables`, `shadows` | engram row tables when on host; V4.1's window shadows, pinned (`deepseek41/kv.rs` module doc) | the same bytes, in the pool | fixed | placement (role rule) |
| `arena` (the NVMe tier's slots) | `AnonMap`; `NVTIER_BYTES` or the H/2 rule; r4b shape 21.47 GB, S ≈ 141 | `cuMemHostAlloc` WC, counted as card bytes ((h)) | budgeted, elastic down to `ArenaFloor` 2B+1 = 33 slots a layer (`:2686`) | placement sizes it; the engine fills it |
| churn pool (adaptive residency's host share) | the card's experts past `pinned` (`churn.rs` header: "the plan's host headroom … must take the pool's bytes"); 0 on a paged stack | 0 (no card mover, (h)) | budgeted | placement |
| KV pool (sealed states + pinned staging) | states pageable (`SeqState::save`), the staging slab pinned once at load at the cache's budget (`checkpoint/staging.rs:5-9`); budget = half the room past the need, floor two states at the plan's ctx (`placement.rs:2455-2459`; the floor is unified-only today) | the same, pool bytes; the save is a pool → pool copy ((l)) | budgeted, LRU-evicted | placement's `within_cache` (`placement/ctx.rs:129`); the pool `checkpoint/{saved,staging}.rs` |
| rings, handoff pages (`MappedHost`) | tens of MB | the same | fixed | the lane / the model |
| page cache (the mapping's file-backed pages) | after S8: the dense tensors and host segment the load read; reclaimable | the model file's pages after load; reclaimable | reclaimable; asserted, not managed | the kernel; the census prints `page_cache_bytes` ((d)) |
| `os` | 6.69 GB | the pool's own margin (#16) | fixed | placement |

The split: fixed terms first, then the two-state KV floor, then the arena and the KV pool share the rest by the dial
(today the cache takes "half of what the room leaves past the need", `placement.rs:2455`). The target makes the dial a
term of one solver ((l)) with this **reclaim order** when a budget must grow at serve time (a `--ctx` past the plan, a
second slot): **page cache → KV pool states (LRU, down to the two-state floor) → arena S (down to `ArenaFloor`) → refuse
by name**; the host segment and the card segment never move at run time (a fault per use, or a re-plan). The page
cache's limit is by construction, not a knob: after S8 no reader faults a paged id, so the cache holds only what the
load read, and the `MemAvailable` reading already counts it as room — the kernel reclaims it under the arena's and the
pool's allocations (#13 measured exactly this: allocations went on to 111.7 GB "as the kernel gave page cache back").
`page_cache_bytes` beside `HostNeed` per span is pinned to ≤ dense + host-segment bytes + 2 GiB [assumption] once
`HOST_LOCK` is on; before that it is printed only.

Numbers (Qwen3.8 UD-Q4_K_XL, 111.3 GB): one sealed state = 28,416 B a position + 260.3 MB of checkpoints [derived from
`qwen38_state_bytes`'s pinned formula, `qwen35moe/place.rs:2267-2300`: 12 attention layers × (2,304 + 256/4) B a
position; 2 × (36 × 3,596,288 + 696,320) B] → 1.19 GB at 32K, 3.98 GB at 128K, 7.71 GB at 262K [the last pinned by
the test]. Box: the two-state floor 2.4 / 8.0 GB against a 263 GB room never binds (the 27G room is the gate dial, not
the box). GB10: 125.2 − 111.3 = 13.9 GB for KV and runtime [derived, dgx-spark.md §5] → at 128K one slot's state 3.98
+ the floor 7.97 = 11.95 GB fits; at 262K 7.71 + 15.42 = 23.1 GB does not — the cache-bounded default context
(`f51eea22`: a Spark's Q4 default 198,144 positions) is this arithmetic, already in the tree.

### (h) Unified memory: what residency and promotion mean on GB10, where the arena sits

Facts [measured, #13 comment 6102768921's follow-up, outside a lease, exploratory]: a kernel reads `cuMemAlloc` at
233 GB/s and `cuMemHostAlloc(PORTABLE|DEVICEMAP|WRITECOMBINED)` at 235 either way (first read after a fill, or repeat);
the same flags without WC 223 repeat / 199 first; an anonymous mapping 157 (2 MiB pages 159; `cuMemHostRegister` on it
moves nothing: "the rate follows who allocated the pages"); `O_DIRECT` reads land in WC pages byte for byte at the
drive's rate (11.20 vs 11.16 GB/s anonymous); a CPU write then a kernel sum agree with no fence over 2,000 rounds; a
CPU streaming copy out of WC runs 27.8 GB/s (small random reads not measured). Pool peak 273 GB/s. Cited: "On
integrated GPUs (i.e., GPUs with the integrated field of the CUDA device properties structure set to 1) … mapped pinned
memory is always a performance gain because it avoids superfluous copies"
(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html, "Zero Copy"); `CU_MEMHOSTALLOC_WRITECOMBINED`:
"WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be
read efficiently by most CPUs" (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html, cuMemHostAlloc).

Decision — one policy, the memory kind as its parameter:
1. **The tier states collapse from four to two on GB10.** `Card`, `Arena` and the host segment are one state, `Pool`
   (a slot the card reads in place at 235 GB/s = `cuMemAlloc`'s 233 for sequential rows, and a random slot gather
   keeps that rate only with whole-line loads over 128-B aligned parts, §6.2 row 1; a copy between them saves 0 bytes
   a use and costs 2 × 3.134 MB of pool traffic a fill); `File` stays. Card residency on GB10 *is* arena residency: a
   layer's "card experts" are the slots of the one arena, S_pool = ⌊(pool − dense − KV − floor) / slot bytes / layers⌋; the card
   mover is `off` with the why `Unified` (no second level to move to); the book's heat ranks eviction (arena → file)
   and the lookahead's prefetch (file → arena) only. Promotion is the fill itself. The step port's leg is the card leg
   (`ArenaLeg`, `crates/gpu/src/host/arena_leg.rs`). The host holds a service's arena ids filled and pinned
   (`NvTier::hold_for_card`, until the row's next go) and writes their parts' word offsets in the arena
   (`NvTier::card_offsets`) beside the handoff. After the row's wait the card computes them with the `_at` entries,
   the `_sel` shapes that read an expert at word offsets of the arena (gate and up in Q4_K or Q5_K, down in Q5_K or Q6_K),
   and adds their weighted sum to the row's `hsum`. The card quantizes the activation as q8_1 and sums in its own
   order, so its steps are not the host leg's bit for bit. The entries are the one kernel-side addition, per quant,
   never per model. A load keeps the host leg when a covered layer's types have no entry, a part starts off a
   4-byte word or a layer has fewer slots than a pass's rows hold ids, when its arena is anonymous (the card has no
   address) and when its residency machine runs (it frees slots a hold keeps).
2. **The arena is `ArenaMem::Pinned`: one `cuMemHostAlloc(PORTABLE|DEVICEMAP|WRITECOMBINED)` for the whole tier**,
   carved by `layer_bases` as the anonymous arena is; Q18 planned one per `LayerArena`, and §6.2 row 2's 64 GiB success
   makes one enough. The `Registered` variant goes (157–160 GB/s registered or not [measured, #13]: worth 0). On
   discrete the arena stays `Anon`: the CPU reads it 480 picks a token, WC "cannot be read efficiently by most CPUs",
   and the 27.8 GB/s streaming figure says nothing about qdot's 32-B row walks. A discrete card's `Pinned` arena, when
   the lever asks for one, is allocated without `WRITECOMBINED`. `give_back` is a no-op on `Pinned` ((d)): the arena is
   a fixed pool allocation priced at load (a linear cost in its size, §6.2 row 2), counted in `HostNeed.arena` as the
   anonymous arena is. The card leg leaves it there: on one pool `PoolOver` sums the cards' and the host's terms, so
   moving it to `HostNeed.cards` / `pool_card_bytes` (`placement.rs:859`) would not change the sum. A
   `MADV_DONTNEED` would make the next fill fail ((d), D1). A fill overwrites the slot's windows; the DMA write of 3.134
   MB into the pool is 13 µs of the 235 GB/s [derived], paid once, no copy after it.
3. **The pool's budget is one number.** #16's `PoolOver` (Σ card + host ≤ `MemAvailable`) stays the refusal; the
   arena's `cards` term makes the (g) table's "arena" and "card experts" rows one row on GB10. A model above the pool
   (GLM UD-Q4_K_XL 199.7 GB, V4.1 Q3_K_M 347 GB) is the only reason the NVMe tier exists there; a model inside it
   (Qwen3.8 UD-Q4_K_XL 111.3 GB) plans every expert into `Pool`: S = all, 0 fills.
4. **Per fill on GB10** [derived from the drives of §6.2 row 3]: 3 parts of ~1.045 MB at QD 3 ≈ ~~7.3 GB/s (Q1 3.73 → Q4
   8.29, interpolated) = 0.43 ms~~ 4.57 to 5.45 GB/s [measured] = 0.58 to 0.69 ms [derived] ([#13 k-rows][krows]). That
   is one fill's share of a 48-part batch read at QD 3; a fill of 3 parts timed alone took 0.61 to 1.34 ms (§6.2 row
   3). At the 11.2 GB/s plateau a fill takes 0.28 ms (#13's 1 MiB loop at QD 16 to 32 on its 4 TB drive, not a decode
   fill's QD 3). The CPU's part is three preads (≈ 30 µs [assumption, 10 µs a pread]). A hit costs the card 3.134 MB /
   ~~235~~ 241.4 GB/s = 13 µs an expert [derived], read with whole-line loads over 128-B aligned parts ([#13
   k-rows][krows]; today's slower gather is in §6.2 row 1). On GB10 the card leg takes a decode step's arena ids
   off the host leg, so the host leg's 23.6 µs and (c)'s host-CPU row go for those ids; the plan's host segment (the
   experts the file mapping serves) and a prompt's batch stay with the host leg. Above the pool the decode wall is card bytes at 235 (slot hits ~~235~~ 241.4, as above) +
   fills × a fill (0.28 ms at the plateau, ~~0.43 ms~~ 0.58 to 0.69 ms at QD 3): GLM on one Spark, 88 GB paged [spark2x
   capacity row], at the box's gate-plan miss rate (45.7 fills a token [measured]) adds 13 ms at the plateau and ~~20
   ms~~ 26.5 to 31.5 ms at QD 3 [derived: 45.7 fills × a fill's batch share], up to about 61 ms with the fill timed
   alone [derived: 45.7 × 1.34], to a ~50 ms step [derived; the Spark's own miss rate is (k)'s ask #4, §6.2 row 4].

Measured since (§6.2 rows 1 and 2): a `_sel`-shaped gather (random slot rows, 32-B strided) from WC pages against the
sequential `hostread2` rows kernel, and the time of one ≥ 60 GB `cuMemHostAlloc`.

### (i) How the GPU reaches NVMe: host bounce, GDS/cuFile, or the in-place read

Availability, by machine [cited; fetched 2026-10-11]:
- GDS mode: "Data Center and Quadro (desktop) cards with compute capability > 6" are supported; "All other cards are
  supported only in compatibility mode" (https://docs.nvidia.com/gpudirect-storage/release-notes/index.html, §5 Support
  Matrix) — the A6000 (Quadro-class, cc 8.6) may, the 3090 (GeForce) may not. The direct path rests on "upstream
  kernel PCI P2PDMA infrastructure for x86_64 platforms" or `nvidia-fs.ko`
  (https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html, §1.3.2), and since v1.7.2 "Proprietary NVIDIA
  kernel module is not supported. Only the NVIDIA open kernel module will be supported" (release notes §3); the box's
  module flavour is [unverified].
- Compatibility mode: "A compatibility mode is available for unsupported configurations that maps IO operations to a
  fallback path" (overview §1.2.3): POSIX reads into host memory, then a copy to the device.
- GB10: "On DGX Spark, GPUDirect Storage is supported only in compatibility mode" and "Do not load `nvidia-fs`, because
  doing so can result in errors" (release notes §11 Known Limitations).

Per expert (Qwen3.8: 3.134 MB a fill, 3.16 MB a slot) [derived from the measured rates above]:

| Path | Drive bytes | DRAM / pool passes | CPU bytes copied | CPU time | Latency until the leg can read it |
|---|---|---|---|---|---|
| A. host bounce, discrete, `Anon` (today): `O_DIRECT` → slot; a promotion stages slot → ring (memcpy) → H2D | 3.134 MB | write 3.134; a promotion reads 3.16 + writes 3.16 | 3.16 MB, promotions only | 3 preads ≈ 30 µs [assumption]; a promotion's memcpy at ~10 GB/s one thread ≈ 0.3 ms [assumption] | fill 0.64 ms (QD 3) + promotion 0.3 + H2D 3.16 / 26.28 GB/s = 0.12 ms |
| A′. as A, `Pinned` arena (§8 Q5): the lane DMAs from the slot | 3.134 | write 3.134; the DMA reads 3.16 | 0 | 30 µs | 0.64 + 0.12 |
| B. GDS/cuFile direct, A6000 only: drive → VRAM | 3.134 | 0 | 0 | the cuFile call | 0.64–1.25 ms, paid per card use |
| C. GB10 in-place: `O_DIRECT` → WC arena; the SMs read it | 3.134 | write 3.134 | 0 | 30 µs | a fill: 0.28 ms at #13's 1 MiB plateau; at QD 3 ~~0.43 ms~~ (measured: §6.2 row 3); nothing after |
| C′. GB10 cuFile compat: POSIX read → bounce → copy into `cuMemAlloc` | 3.134 | write 3.134 + read 3.134 + write 3.134 | 3.134 MB | the read + the copy | ≥ C + 3.134 / ~100 GB/s ≈ C + 0.03 ms, plus page-cache bytes (no `O_DIRECT`) |

Decision: **host bounce on discrete, in-place on GB10, one code path — `O_DIRECT` into the arena's slot, and the leg
is the slot's reader** (`TierLeg::Host` with qdot, `TierLeg::Card` from `device_slot`). B is refused by arithmetic,
not availability: on discrete the paged expert is computed on the host, so its slot exists in RAM anyway, and a card
promotion's source is that slot (0 drive bytes, 0.12 ms H2D); GDS would replace a 0.12–0.42 ms copy with a 0.64–1.25 ms
drive read, and need a feature the 3090 lacks. C′ is refused: one more pool pass and a page-cache footprint (g) would
have to manage. cuFile enters nowhere; the one new seam is Q18's `device_slot`.

### (j) Read granularity, queue depth, prefetch depth, bytes in flight

The measured curve is already per part: qdsweep's `directN` arm runs Q workers, each one blocking `pread` of a
~1.04 MB part at a time (`docs/cards/qdsweep.card:6-9`), so R(QD) = 2.51 / 4.15 / 5.67 / 6.48 / 6.77 / 6.72 GB/s at
QD 1/2/4/8/16/32 [measured, 10-11] is the part unit's curve on the box; the Spark's at 1 MiB is 3.73 / 8.29 / 11.16 /
11.20 at Q 1/4/16/32, 256 KiB at Q16 9.96, 4 MiB at Q16 10.78 [measured, #13]. The block layer's cut differs: the box
splits at `max_sectors_kb` 1280 (`qdsweep.card:18`), so a part is **one** NVMe command and QD 1 is one command in
flight; the Spark splits at 128 KiB, so a 1 MiB pread is 8 commands and its Q1 already reads 3.73 [#13]. Alignment: a
part's file offset is 32-B aligned (Q18 §1) and `aligned_span` (`engram/src/direct.rs:132`) rounds to 4 KiB, so a part
read wastes ≤ 2 partial pages = ≤ 8 KiB, 0.8 % of 1.045 MB [derived]; a 1/8-part tile would waste the same 8 KiB on
131 KB (6 %) and cost 24 syscalls a fill.

| Choice | Rule | Box | Spark |
|---|---|---|---|
| unit | the part (3 a fill), as (b) and S2 | 1.045 MB = 1 command; a fill at QD 3 ≈ 4.9 GB/s (4.15 → 5.67 interpolated) = 0.64 ms vs 1.25 serial [derived] | 1 MiB = 8 commands; QD 3 ~~≈ 7.3 GB/s = 0.43 ms [derived]~~ (measured: §6.2 row 3) |
| sub-unit (S2b: only if S2's measured QD-3 fill is ≥ 0.6 ms) | part / 3 ≈ 350 KB, so one fill carries 9 units ≥ the knee | QD 9 ≈ 6.5 GB/s: 0.48 ms, −0.16 ms a fill = −7 ms a token at 45.7 fills (−8 % of 85) [derived]; +6 preads ≈ +60 µs CPU across threads; waste 2.3 % | ~~≈ Q9 ~10 GB/s [interpolated Q4 8.29 → Q16 11.16]: 0.31 ms, −0.12 a fill~~ (derived again from the measured rates: §6.2 row 3) |
| queue depth (reader threads) | the plateau's QD | 16 (6.77; QD 8 is 96 % of it) | 16 (11.16; Q32 11.20), #13's 1 MiB loop; parts at QD 16: §6.2 row 3 |
| background share during steps | Victim + Prefetch ≤ the knee, so a Step's 3 units take threads at once | ≤ 8 threads: a Step unit never waits a background unit's sojourn (16.7 MB / 6.77 GB/s = 2.47 ms at a full QD 16) | ≤ 8 |
| prefetch depth, decode | one layer ahead: a layer's compute (37 ms / 48 = 0.77 ms at the gate plan) ≥ a QD-3 fill (0.64) hides it; two ahead hides nothing more and doubles the (1 − r) waste [derived] | 1 layer, ≤ 3 × misses units (≈ 3 at 0.95 miss a layer) | 1 layer |
| prefetch depth, prompt | one batch ahead (built): 16 picks = 48 units = QD 16 | 48 units | 48 units |
| bytes in flight (Little: L = N / R) | at the plateau | 16 × 1.045 = 16.7 MB; a unit's sojourn 2.47 ms (the prompt's regime); an idle-queue Step fill sees QD 3: 0.64 ms | 16.8 MB; 1.5 ms (both at #13's 1 MiB plateau); Step ~~0.43 ms~~ (measured: §6.2 row 3) |

This amends (b)'s "shares, not reservations" by one number: a background class holds at most 8 of the 16 threads while
a step runs, and any thread between steps (the prompt batch is all 16). The per-drive parameters (`max_sectors_kb`,
r1, R_p, the knee) are read at load from sysfs and the drive's card (`tools/ref/card.py`'s curve), never a constant, so
the unit and the shares are one code on both machines.

### (k) N-node Spark (ConnectX-7 ×2/×4): where the books and the scheduler sit

Link facts [cited; https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html, fetched 2026-10-11]: "Each DGX Spark
has two QSFP ports (sometimes called “ConnectX-7 ports”) on the back of the device"; "Each port provides up to 200
Gigabits per second (Gb/s)"; the NIC reaches the SoC "through two independent PCIe Gen 5 x4 links"; "It supports up to
three DGX Spark systems connected directly through cables, and up to four systems when using a switch"; "The QSFP ports
support Ethernet configuration only". No GPUDirect RDMA on GB10 (the porting guide, quoted in `docs/contrib/dgx-spark.md`
§9.1): the NIC reaches only `cudaHostAlloc` pages — which the WC arena *is*. One cable measured 189.85 Gb/s = 23.7 GB/s
a direction, one rail and QP 92.57 Gb/s = 11.6 GB/s (NVIDIA's playbook, dgx-spark.md §9.1). λ (one round trip) is ~~unmeasured~~
measured on two DGX Sparks ([#44][lambda]; the value is in `docs/contrib/dgx-spark.md` §9.1); spark2x derives 25–70 µs.

Decision — **a per-node engine over a static placement split** (spark2x Q4's shape), by three numeric criteria:
- **C1, claims are local.** Books per node, `(node, layer, id) → state`, each node its own `NvTier` over its own pool
  arena, drive and file copy. The alternative, one global book on node 0 with remote claims, puts λ on every claim: 480
  claims a token × 2 messages × 25–70 µs = 24–67 ms against a 20–50 ms step → refused for every N ≥ 2. The rule: λ ×
  remote claims a token ≤ 1 % of the step (≤ 3–12 claims a token), which only a local book meets.
- **C2, no remote-arena tier.** The I/O scheduler is per node: a node's misses go to its own drive (a 4 TB Gen5 part
  each, 11.2 GB/s plateau each — drives scale with N, rails do not). Drives differ between units: 11.2 GB/s is one
  unit's 4 TB Gen5 drive ([#13 drive][drive13]), and a 1 TB Gen4 DRAM-less unit reads less ([#71][drive71], §6.2 row 3).
  Node j's pool as node 0's slow tier (a fetch of 3.134 MB a use at ~~11.6 GB/s + λ = 0.30–0.34 ms~~ the measured
  13.63 GB/s plus λ_probe's p50 of 30 µs = 0.26 ms [derived, [#13 k-rows][krows]; `docs/contrib/dgx-spark.md` §9.1])
  pays only when the id is reused ≥ ~~0.30 / 0.013 ≈ 23~~ 0.26 / 0.013 = 20 times before node 0 evicts it, since
  node j computes it in place in 13 µs + λ — a hot id the split already homes where it is used. ~~The NVMe plateau
  (11.2) equals one rail (11.6)~~ One rail moves 1.22 times the 11.2 GB/s drive plateau [derived,
  [#13 k-rows][krows]]: the link is not a faster drive.
- **C3, balance includes fills.** The window per layer is the max over nodes of (its slots × 13 µs + its misses ×
  a fill (0.28 ms at the plateau, ~~0.43 ms~~ 0.58 to 0.69 ms [derived] at QD 3; (h) item 4) + λ for a remote); a
  node with the colder set bounds the step, so the balancer's objective gains the predicted fills per node within 10 %
  of each other (spark2x's S2-ii extended to fills) beside S3's byte balance.
- Heat stays one source: node 0's tally ships E × L u32 = 512 × 48 × 4 B = 98 KB a window [derived] to each node for
  its eviction; the cross-node mover (Pool@i ↔ Pool@j, 3.134 MB a move = ~~0.27 ms at 11.6 GB/s~~ 0.230 ms at the
  measured 13.63 GB/s [derived, [#13 k-rows][krows]] of NIC, off the step) rebalances on heat drift under S4 (NIC
  ≤ 50 % of the step: ≤ ~~74~~ 87 moves a token at a 40 ms step [derived, [#13 k-rows][krows]] — never binding). The
  split is static per load (each paged id has a home node from the byte ledger, S2-i per node); the runtime moves only
  inside a node's budgets; a home change is the mover's, never the plan's.

The six Spark-only facts the design could not derive went to a contributor; the asks, their results and the
decisions they changed are in §6.

### (l) KV and placement in one policy

The lead's cut is accepted, with one derivation fixing what the runtime may move. **Placement is the load-time solver
over one memory model** — per device `{kind: Card | Host | Pool, capacity, bandwidth, reclaim order}`, on GB10 one
`Pool` device — whose output is every budget and (g)'s reclaim order: card experts (or S_pool), the KV pool at ctx ×
slots × state bytes, the arena's S, the prompt cache's budget and floor, the checkpoints' host room, the churn pool, the
page cache's assertion, the OS reserve. `expert_nvme_tier`, `ArenaFloor`, `grow_kv` (`placement.rs:814`),
`within_cache` (`ctx.rs:129`) and the paged cache default (`paged_default_cache_is_off_by_the_split`,
`gate_qwen38_serve.rs:3593`) become terms of that solver; the case-by-case rules go. **The tier engine is the runtime**:
it moves expert slots (promote, demote, prefetch) and KV *states* inside those budgets, on one account (the budget
ledger the census prints) and one scheduler per resource — the reader for the drive, the save lane for card ↔ host
copies: two executors, one ledger. A budget change goes back through placement: a serve-time `--ctx` past the plan runs
(g)'s reclaim order, then refuses by name.

**What the runtime may move, by derivation.** KV moves as whole sealed states, never as an active sequence's older
blocks:
- An active sequence's attention reads every position each token. Qwen3.8's 12 attention layers hold 28,416 B a
  position [derived, (g)]: 0.93 GB at 32K, 3.72 GB at 128K, read on the card in 1.6 / 6.2 ms a token at ~600 GB/s
  [assumption: the A6000's gemv rate]. A fraction f of those positions on the host costs f × 0.93 GB / 26.28 GB/s =
  f × 35 ms a token at 32K and f × 142 ms at 128K over PCIe [derived] against a 32 ms step: **it never pays on
  discrete** at any f worth the machinery. V4.1's selector changes nothing: its index keys are read whole each token
  and the selected latent rows change per token (a scattered gather, latency-bound). On GB10 there is no bus and no
  "older block": the KV is pool bytes wherever it sits.
- A sealed state (a session the server switched away from) is read only on resume: the KV pool v2
  (`checkpoint/{saved,staging}.rs`, `SaveLane`) already moves it card → host RAM on discrete; its second tier is NVMe
  when host RAM is short. A 32K state (1.19 GB) restores from the box drive in 0.18 s at 6.77 GB/s, a 128K state
  (3.98 GB) in 0.59 s, against a re-prefill of 128K tokens at ~1,000 tok/s ≈ 131 s [derived; the pp rate an
  assumption] — ~200×; on the Spark 0.36 s at 11.2 GB/s. This is `Kind::State` in the reader (rank 4, below Prefetch;
  1 MiB units; QD 16 = the plateau), the state's file in the cache's directory — the one job class the KV side adds, so
  the drive's QD stays one account with the experts' fills.
- On GB10 the save copy itself is the lever: a 32K save is a pool → pool copy of 1.19 GB (≈ 10 ms at ~120 GB/s
  [assumption]); a sealed state can keep its rows in place and the slot's KV region be re-pointed (0 bytes). Small; a
  later lift, named so the KV owner's `Pool` kind does not bake the copy in.

Budgets at 32K / 128K [derived]: a slot's card KV 1.19 / 3.98 GB (Qwen3.8); the two-state floor 2.4 / 8.0 GB; on the
box's 263 GB room neither binds; on GB10's 13.9 GB past the file, 128K fits one slot with the floor (12.0 GB) and 262K
does not (23.1 GB) — `f51eea22`'s cache-bounded default is the solver's first term, already in the tree. GLM's latent
layers hold 1,600 B a position each (`glm5next/place.rs:909`: a 1,536-B row + a 256-B pool row per four positions)
[derived; × the trunk's latent-layer count, not read here]; V4.1's per-layer function is `docs/v41-placement.md:108`
[derived there].

Owners (the §2 table's new row): the budget — placement (`within_cache`, the floor at `placement.rs:2455`, `grow_kv`,
each card's KV term in `CardTotals`); the pool — `crates/gpu/src/checkpoint/{saved,staging}.rs` (`SeqState`,
`SaveLane`, `Staging::pinned` at load); per family only the state's composition.

## 4. The cost model

Per token unless named. r4b shape = A6000, plan a, ctx 3072, `BLOOMERY_HOST_ROOM=27G` + `BLOOMERY_NVTIER_BYTES=20G`
(arena 21.47 GB, paged 31.86 GB ≈ 10,164 ids ≈ 212 a layer, S ≈ 141), prose 512, n 96, residency `mid-p0-s1`, draft
off, one slot. Gate plan = the 3090's bytes planned on the card held (`gate_card.rs`): paged 61.49 GB ≈ 409 a layer,
S 154, keep 122.

| Term | r4b shape | gate plan | Tag |
|---|---|---|---|
| step floor (no fills) | 31.7 ms (arena-0 pass) | ≈ 37 ms = 26 card + 11.3 host | r4b [measured, plan-triage.md:277]; gate [derived] |
| card SMs | ≈ 26 ms | ≈ 26 ms | [derived, r4dc §3]; gate [assumption] |
| host compute | 136–238 slots × 23.6 µs = 3.2–5.6 ms | 480 × 23.6 µs = 11.3 ms | W [measured, H8 replay, design Q6]; slots [measured, 09-30 q38seed table] |
| fills a token | cold 54 (5,194 / 96); warm ≈ 0–1 | 87.5 before R4b → 45.7 after (8,400 → 4,385 / 96) | cold [measured, R1 probe]; warm [derived from +154 %]; gate [measured, gate_nvtier real, untimed] |
| one fill | 1.0–1.1 ms = fault/zero 0.35–0.5 + drive 0.6–0.75 | same | total [measured, R1 probe]; split [paper, r4dc1] — inconsistent with directN's QD 1 1.25 ms: the refill card decides |
| drive rate R(QD) | 2.51 / 4.15 / 5.67 / 6.48 / 6.77 / 6.72 GB/s at QD 1/2/4/8/16/32 | same | [measured, qdsweep 10-11, r4/decisions.md:13] |
| fill floor at the knee | 3.134 MB / 6.77 GB/s = 0.46 ms | same | [derived] |
| part-unit fill (QD 3) | 0.63–0.75 ms | same | [derived: 3 × ~1.05 MB at the interpolated 4.2–5.0 GB/s] |
| chain fill wall | ≈ 0–1 ms (warm); 55 ms (cold) | 46–50 ms | [derived] / [measured, R1 probe] |
| flips, victims | 5.75 a token, 553 jobs / 96 steps | 552 flips, late 0 | [derived r4dc] / [measured r4da] |
| victim wait | 2.41 ms a job (= 1 unit: give-back + refault + 3 serial reads) | — | [measured r4da]; body split [paper, r4dc1] |
| lane window vs demand | 2 passes × 32 = 64 ms vs F × 2.8 ms, F 11.5–23 → 32–64 ms (near its bound) | 2 × 85 = 170 ms, far from it | [derived r4dc §3] |
| PCIe | 5.75 × 3.16 MB / 26.3 GB/s = 0.7 ms, async | same | [derived] |
| lookahead cost | router 0.33 ms + 48 launches 0.04 ms, no extra bridge | same | [derived; router type assumption] |
| lookahead gain at r | r × remaining chain fills − 0 (waste is off-chain while drive time < token) | at r 0.7 after bulk+refill: −17 ms | [derived]; r [assumption 0.6–0.85] |
| off-rule first term | Δh × 136 × 23.6 µs = 1.66 ms (Δh 0.517) ≫ 0.32 (1 % of floor) | 0.517 × 480 × 23.6 = 5.86 ms | [derived; Δh measured on the 3090 replay] |
| landing give-back | 5.75 × 70–100 µs = 0.4–0.6 ms | same | [assumption 80 ns a page] |
| ruler | sd 2.6 % (the residency pair); 6 rounds h_p 3.86 %, 10 rounds 2.63 % | — | [measured, rig-log 09-30#q38seed-ab; card.py] |

The decode ladder at the gate plan [derived]: 85 → 66–71 (part units) → ≥ 58 (+refill, bounded by the rate floor) →
48–58 (+in-layer overlap) → 40–48 (+lookahead at r 0.7) → 37. On the r4b warm window every fill term is ≈ 0 and only
the long-decode curve (the second open term) says whether the A6000 shape ever leaves the floor.

## 5. The migration plan

Each step lands alone (express) unless named; every step's A/B runs at a shape with fills (gate plan or n ≥ 1024) or
is refused by its own card. "Deletes" is what leaves the tree in that step or the next.

| # | Step | Class, proof | Gates, FAIL-first | The one box measurement, expected | Deletes | Status (main `9bd7d577`) |
|---|---|---|---|---|---|---|
| S0 | **Records by kind + the runner keeps arm stderr** (f): `KindStats.run_ns`, `{step,keep,transient}_*`, `fill_{fault,read}_ns`; `depth-qwen3moe.sh`/`ab-decode.sh` write `<log>.arm<n>.err` and fold `nvtier`/`residency` into the row | records (Kind change → `records-refresh`); host only, ptx-scan identical | `gate-gpu-nvtier` kind test (re-pinned with its derivation); runner self-test: an arm without stderr → red | none (a functional gate run prints the keys; the fault/read split is read on the next card as a beside field) | `count_fill`'s `pinned` key; the arm-record gap named in r4d-decode.card and r4dc-default.card | Partly in: records by kind are in (`967f083f`, `3301f20c`); each arm's `nvtier` record reaches `depth-qwen3moe.sh`'s log (`44b2d409`); the fault/read split and the per-arm stderr file are not |
| S1 | **refill**: delete the pre-read `give_back` in `read_pick` (`:2243`, `PickRead.give_back`, `Pick.gone`'s use); keep `release`'s | host dispatch path → same-lease A/B (refill's card, at the gate plan or n ≥ 1024) | `gate-gpu-nvtier` bit identity (same bytes, the audit); FAIL-first: a short read must still refuse by name (`:2252`) | fill 1.05 → 0.6–0.75 ms, −16..−23 ms a token at the gate plan (+19..+37 % tok/s); at the r4b warm window inside the ruler → refused | the refault; `AnonMap::dontneed` keeps one caller | In (`af6d0eba`); its card read +7.07 % ± 1.61 % (95 %, 10 rounds) at n=1024, residency off, on the A6000 paged plan (27G + 20G), not the gate plan (`docs/cards/refill-decode.card`: main `b00ff641` against `b00ff641` with `af6d0eba` reverted; the result in `23ee2100`) |
| S2 | **One reader, part units, `Kind::Step`** (b): the step's misses submitted to the reader at rank 0, the chain waits on the latch; unit = part; Victim/Prefetch take a thread only while Step is empty | host dispatch path; ptx-scan identical; the reader's count identity `fills == step_units/3 + prompt_units/3 + victim_units/3` | `gate-gpu-nvtier`: a new clause "the pool makes no NVMe read" (`step_jobs > 0`, pool fill count 0); FAIL-first: route Step through the pool → red; a Victim ahead of a queued Step → red | step fill 0.63–0.75 → net with S1 ≥ 0.46 ms; −14..−19 ms a token at the gate plan on top of S1 (bounded: S1+S2 ≤ −27 ms); the hand-off 0.05–0.15 ms a job is the term to watch (noninf margin 5 %) | `NvTier::fill` (pool path), `ensure`/`ensure_as` sync forms, `fill_victims`' own submit (a victim is a job like any), the lane's blocking under the receive lock while the reader runs (the lane submits, then stages when the latch opens) | Not in main |
| S3 | **In-layer overlap**: `experts_into_on` computes the hits while the Step job is in flight; per-expert outputs stored, summed in list order | host path; bits identical by construction (same order of the final sum); proof: `gate-gpu-e2e` tokens equal | `gate-gpu-nvtier` steps clause; FAIL-first: sum in landing order → the bit pin red | −5..−10 ms a token at the gate plan (0.21 ms × layers with a miss); 0 at the warm window | nothing (the fill's join moves after the hits' compute) | Not in main |
| S4 | **Off rule derived** (e): `GainUnderRuler` in `room_for`, the `Paged` clause's 36 sites go (r4dc2), one `RoomShort` → each family's why | host plan path; levers unit tests; proof: the probe table of r4dc1 §2 reproduces (plan a → `mid-p0-s1`, GLM 3090 → `off` NoCardExperts) | levers tests (FAIL-first: restore the clause → the paged-shape pin red); `gate_nvtier` clause 5 re-pinned to the word; `nvtier_family` per-family pick | r4dc-default.card as written (−2..+10 %, 10 rounds, the flip's own A/B) | `PlanTier`, `PagedText`, `ResidencyWhy::Paged`, `Residency38Why::Paged`, `GlmWhy::Paged`, `RoomShort::Paged` | The `Paged` clause, `PlanTier`, `PagedText` and `RoomShort::Paged` went with R4d-c (`967f083f`); `GainUnderRuler` is deferred (§8) |
| S5 | **Gate lookahead** (c): probe first (a `lookahead` record, functional, no lease); then the round: router_{l+1} on the pre-attention residual in the card graph, k ids on the handoff page, `Kind::Prefetch` jobs cancelled by a landing Step claim | add class on the card (ptx-scan = base + the router entry, `gate-ptx-spill` red on it before pinning) + host path; launch count Δt = 48 × c_node | `gate-gpu-e2e` (bits: the real router unchanged); new nvtier clause `prefetch_hit + cancelled + wasted == issued`; FAIL-first: a prefetch that claims a pinned slot → red | the probe: r at k = m, expected 0.6–0.85; the round's A/B at the gate plan: −10..−20 ms a token at r 0.6–0.85 after S1–S3; refused below r 0.5 | nothing until measured; at r < 0.5 the step is dropped and (c) closes with the three slot-level levers | Not in main |
| S6 | **Heat in the book** (a): one count per (layer, id) written by the tally, read by the rule and `pick_slot`; `SwapRule.counts` a view | move; proof: a Mac route-trace replay plans identical flips from both sources; `ptx-scan` identical | `gate-gpu-qwen38-residency` / `gate-gpu-swap` flips pin; `gate-gpu-nvtier` eviction clause (coldest count, transient first); FAIL-first: evict by tick alone → the clause red | none (counting identity); read beside: fills a token at n 1024 (expected to fall with count-ranked eviction vs tick LRU, unmeasured) | `SwapRule.counts` as a second copy; `NvTier::Heat`/`slot_heat` (no caller) or they become the owner | Not in main |
| S7 | **`ArenaMem::Pinned`** (d): the enum, `give_back` no-op on Pinned, `device_slot` for an integrated read, `devmap` clause checked on any card ([#13 devmap][devmap]) | add behind `BLOOMERY_NVTIER_MEM` (`auto` = `pinned` on an integrated device, `anon` on a discrete card; a pinned arena is write-combined on an integrated device only) | Q18's clause as two `devmap` hw tests in `nvtier.rs`: DtoH of `device_slot` == `slot()` bytes; FAIL-first D1 (dontneed on Pinned → the next fill's read fails, red on an integrated device; a PCIe card without write-combining keeps the bytes and stays green, [#109][i109]), D2 (part skew dropped → red) | the one term: `cuMemHostRegister`/`cuMemHostAlloc` over 20 GB works and its time (~~expected 2–4 s [assumption]~~ measured on GB10: both work; `cuMemHostAlloc` 1.59 s at 19.91 GiB (median) and 5.288 s at 64 GiB ([#13 k-rows][krows]), `cuMemHostRegister` 0.543 s over 19.91 GiB of pages already touched (median, [#13 register][measure13]); conditions in §6.2 row 2); bits equal | the integrated arena as a separate plan | In: one allocation for the tier, behind `BLOOMERY_NVTIER_MEM`, whose `auto` is Q5's answer (`pinned` on an integrated device only); (d) and (h) items 1 and 2 describe it. The card leg ((h) item 1) is in for GLM-5.3-Flash's step port |
| S8 | **Dropper and mapping path for paged ids go** | delete; per entry the grep with no caller; `mapping_reads == 0` and `drops == 0` pinned | `gate-gpu-nvtier` clause 9 `exclusive` turns the tolerated mapping reads into a 0 pin (dated); FAIL-first: a lane stage of an unfilled id through the mapping → red | none (a functional run prints the pins) | `nvtier-drop`, `drop_layer`, `end_read`'s queue, `PageDrop` for paged ids, `source()`'s mapping fallthrough for arena ids | Not in main |
| S9 | **Lift the copies** (§2): Qwen3.8's host-run build onto `model.rs:1710`; one `placement::Room` read per `describe`; MiMo's body attach through the same call | move; `ptx-scan` identical; `just affected --narrow` | each family's nvtier gate (`nvtier_family.rs` with a MiMo caller) | none | `body38.rs:1451-1456`'s own build; three room fields | Not in main |
| S10 | **One budget solver** (g, l): a `MemoryModel` per device (`Card | Host | Pool`, capacity, bandwidth, reclaim class) whose terms are card experts / S_pool, the KV pool at ctx × slots × state bytes, the arena's S, the cache budget and floor, the churn pool, the page-cache assertion; the (g) reclaim order at a serve-time `--ctx` | host plan path, pure crate; proof: every plan in `tools/ref/plan-room.sh`'s room prints the same budgets as before (a counting identity) and the four replaced rules' tests re-point to the solver's terms | placement lib tests; `gate-gpu-qwen38-serve`'s two paged clauses re-pinned to the terms (dated); FAIL-first: a reclaim that touches the host segment → red; a cache under the two-state floor → red | none | `expert_nvme_tier`'s room arithmetic, `ArenaFloor` as a free rule, `grow_kv`, the paged cache default, the per-family room fields (with S9) | Not in main |
| S11 | **Reader shares and per-drive parameters** (j): background ≤ 8 of 16 threads during steps; `max_sectors_kb`, r1, R_p, the knee read at load; S2b (part/3 tiles) opened only if S2's measured QD-3 fill ≥ 0.6 ms | host path; the reader's unit tests | `gate-gpu-nvtier` clause: a Step unit's `wait_ns` ≤ one background unit's service time; FAIL-first: a 16-thread background class → red | S2b's own card: −0.16 ms a fill, −7 ms a token at the gate plan (−8 %) | nothing | Not in main |
| S12 | **`Kind::State`** (l): sealed states' NVMe tier behind the pool's `NoRoom`, through the reader at rank 4 | add, host only; `ptx-scan` identical | `gate-gpu-qwen38-serve` clause: a state evicted to NVMe resumes bit-identical (tokens equal); FAIL-first: one page short on the restore → red | a 32K restore 0.18 s (1.19 GB / 6.77 GB/s), functional | the request-path fallback for a sealed state that found no room | Not in main |
| S13 | **N-node** (k): `Machine.nodes` + `Fabric` (spark2x Amendment 1), per-node `NvTier`, the 98 KB heat window, the mover lane under S4's cap | placement first (pure; `nodes_scale` + C3), then the lane; after S7 and the contributor's #5 | `nodes_scale` with the per-node fills criterion; the lane's bit identity on two processes of one box (loopback) | λ on one rail (the contributor's) | nothing in this tree | Not in main. A contributor holds the placement half (`Machine.nodes`, `nodes_scale` + C3; `docs/contrib/dgx-spark.md` §9 and §9.6 item 2), which does not wait for S10 (§8); the lane half follows S7 |

Order and trains: S0 (partly in) → S1 (in) → S2 → S3 → S4 + S10 (one owner of `placement`/`levers`: S10 replaces the same
rules r4dc2's clause removal touches) → S5 probe (free, any time after S0) → S6 → S7 → S11 (reads S2's measured QD-3
fill) → S8 → S9 → S12 → S13. S1+S2 share `read_pick`/`fill`: S2 rebases on S1. S2 and S4 touch `gate_nvtier.rs` and
`nvtier.rs`: one owner per round, S4 after S2. Every step deletes or converts a path within two steps: S2 removes the
pool reader that S1 touched; S8 removes the mapping path S2 made unreachable; S10 removes the four scattered budget
rules the moment the solver prints their numbers. GB10-only behaviour enters as parameters of S7, S10 and S11
(`ArenaMem::Pinned`, the `Pool` device kind, the drive's constants), never as a step of its own. A contributor holds
S13's placement half (S7's card leg is in); the placement half does not wait for S10 ([#13 answer][answer]), and
the lane half stays after S7.

## 6. Measured on DGX Spark

Five of the six Spark-only facts that (k) asked a contributor for are measured ([#13 k-rows][krows]); row 4 is open.
The decisions of (h) and (k) stand, and three inputs change: a hit is priced on whole-line loads, a fill at QD 3 takes
0.58 to 0.69 ms [derived] instead of 0.43, and one rail moves more than the cited 11.6 GB/s. (k)'s rule points the
Spark to S2b's tiles, and S11's gate on S2's measured fill decides (row 3). The results and their conditions are in
§6.2. Where §3's prose used a replaced value, it is struck and corrected in place; §3's tables strike it and point
to §6.2.

Rows 1, 2, 3, 5 and 6 ran on GB10 with driver 595.84, kernel 6.17.0-1032-nvidia, GPU clocks stock (not capped) and
serving off; row 4 has not run. Each row's own conditions are in its last column, and the full setup is in the
linked comment.

### 6.1 The asks, as (k) planned them

The table is (k)'s, moved here. Rows 1 and 2 carry the maintainers' rule corrections of 2026-10-13 (answers 1 and 2 in
[#13 answer][answer]).

| # | Fact | Command (one) | The number decides | We build first |
|---|---|---|---|---|
| 1 | a `_sel`-shaped gather from WC pages: random slot rows, 32-B strided, vs `hostread2`'s sequential rows | `hostread2 2 host-wc` with a `ROWS=random` arm (≈ 20 lines on the comment's kernel; we post the patch on #13) | ≥ 200 GB/s → (h) as written; 160 to 200 → record it and reprice (h) (answer 1); < 160 → each slot row gets a 128-B pad (a plan-bytes change) | nothing: the kernel is the contributor's |
| 2 | one `cuMemHostAlloc(WC)` of 64 GiB: success and seconds | `FILL=$M hostread2 1 host-wc` with a 64 GiB size arg | success → S7 prices the arena at load as a linear cost in its size (answer 2); failure → S7 allocates per layer (48 calls) | S7 (`ArenaMem::Pinned`, `device_slot`) |
| 3 | the part-unit curve on the Spark's drive at our shape: 1.045 MB, QD 3 / 8 / 16, random offsets | `probe_nvread --threads 3,8,16 --file $M --unit 1045000` | QD-3 ≥ 7 GB/s → (j)'s unit holds; < 6 → S2b's tiles on the Spark | a plain-file arm of `probe_nvread` (today it opens a qwen4exp split set, #13) |
| 4 | fills a token above the pool: GLM UD-Q4_K_XL on one Spark, 88 GB paged | `bloomery-serve … --place auto`, S0's `nvtier` record, one 512 prompt + 96 steps | fills × ~~0.43 ms~~ 0.58 to 0.69 ms (§6.2 row 3) against the step: > 20 % → (c)'s lookahead opens on GB10 first | S0 (records by kind), ~~S7~~ (not needed for this reading: §6.2 row 4) |
| 5 | λ and the 3 MiB-message rate on one rail | `ib_write_lat` / `ib_write_bw -s 3145728` between two Sparks (spark2x Q5's first step) | λ ≤ 70 µs → C1/C2 as derived; the move rate sets the mover's cap | nothing |
| 6 | GB10 `c_node` | `cnode_probe --time` (sparkdec §9) | the lookahead's 48 launches a token (0.04 ms on the A6000) on GB10 | nothing: the bin exists |

### 6.2 Results

All results are from [#13 k-rows][krows] unless a row names another source.

| # | Result | Decision | Conditions |
|---|---|---|---|
| 1 | Today's `_sel` gather (4-byte loads, mixed 32-B part offsets) reads random slot rows of a 19.91 GiB WC arena at 167.8 GB/s, 18.7 µs a hit [derived]. `rows_k`'s lane mapping (whole-line loads) over 128-B aligned parts reads 241.4 GB/s on WC and 240.7 on `cuMemAlloc`, 13.0 µs a hit [derived]. Sequential rows read 237.6. | (h) assumes whole-line loads over 128-B aligned parts (241.4 ≥ 200) and keeps 13 µs a hit; 167.8 stays on record as today's gather ([#13 answer][answer]). The < 160 branch's pad does not help today's form (170.7 GB/s with every part aligned, 168.9 mixed). How parts land on 128-B boundaries under `O_DIRECT`'s 4 KiB alignment is not measured or designed yet: the card leg chooses its load shape and part alignment from measurements ([#13 answer][answer]). | 167.8 and 237.6: one Spark (the first in the comment, which also ran row 2's 64 GiB), 6 interleaved rounds, prefetch loop on, memset fill, slots not read since the fill. 241.4 and 240.7: another Spark, one of the two #44 ran on, 3 rounds, prefetch loop off (`PF=0`), arena touched once after its fill, loads only (no decode). |
| 2 | One `cuMemHostAlloc(PORTABLE \| DEVICEMAP \| WRITECOMBINED)` of 64 GiB succeeds in 5.288 s (12.1 GiB/s). The time is linear in size: 1.56 to 2.04 s at 19.91 GiB (median 1.59). One `cuMemHostRegister(PORTABLE \| DEVICEMAP)` over an anonymous 19.91 GiB mapping succeeds in 0.543 s (median, 0.538 to 0.604), 36.7 GiB/s [derived] ([#13 register][measure13]). That time excludes touching the pages first (4.2 to 5.8 s), so it does not compare with `cuMemHostAlloc`'s 1.59 s. | S7 prices the arena at load as a linear cost; 5.288 s is a success ([#13 answer][answer]). No decision rests on the register time: (h) item 2 already drops the `Registered` variant on its read rate. | 64 GiB: one run on row 1's first Spark, filled by `O_DIRECT` reads of a GGUF. 19.91 GiB: row 1's allocations. Register: one Spark (its comment does not say which unit), 3 runs, each a new process; 4 KiB pages (transparent huge pages unused), every page written before the call, so a huge-page mapping is not measured; a driver API program outside the tree, so no bloomery commit. |
| 3 | 1.045 MB parts at random offsets, QD 3: 4.97 and 5.45 GB/s on a 4 TB PCIe Gen5 drive, 4.67 and 4.57 on a 1 TB PCIe Gen4 DRAM-less drive (two sessions each). As one fill's share of those 48-part batches, a fill takes 0.58 to 0.69 ms [derived]; a fill of 3 parts timed alone took 0.61 to 1.34 ms (each run's median). QD 8 reads 8.35 to 8.48 GB/s (Gen5) and 5.91 to 6.01 (Gen4), QD 16 8.4 to 8.9 (Gen5) and 5.8 (Gen4). (j)'s 7.3 GB/s at QD 3 was an interpolation. | Under 6 on both drives, so (k)'s rule points to S2b's tiles (part / 3, 9 units a fill) on the Spark. Nine tiles at the QD-8 rate would take about 0.37 ms (Gen5) and 0.52 ms (Gen4) [derived: 3.134 MB / 8.48 and / 6.01 GB/s; the tile size itself was not measured]. S11's condition (S2's measured QD-3 fill ≥ 0.6 ms) stays the final test, and one Gen5 session's batch rate gives 0.58 ms. | `probe_nvread` built from main `23ee2100`, run through `tools/ref/nvtier-read.sh` under a lease in `box.sh` local mode, on Qwen3.8's real expert parts (k16 batches of 48 parts); page cache dropped before each queue depth; medians of 24 batches. |
| 4 | Not measured yet: fills a token, and fills × the fill time as a share of the step. | GLM loads at the serving default since [#100](https://github.com/midagedev/bloomery/pull/100), but that plan has no arena (`arena=0`), so it makes no fill to count. The reading needs a GLM plan with an arena, which on one Spark takes a lower `BLOOMERY_CARD_BUDGET` beside `BLOOMERY_NVTIER_BYTES` (`docs/contrib/dgx-spark.md` §10). S0's records by kind (`KindStats.run_ns`) are on main, and S7 and its card leg are not needed for this reading: P4 needs them after it (`docs/contrib/dgx-spark.md` §8). | |
| 5 | λ and the 3 MiB rate on one rail are in `docs/contrib/dgx-spark.md` §9.1. With (k)'s formulas, the 13.63 GB/s rail rate and λ_probe's p50 of 30 µs ([#44][lambda]) they give: a 3.134 MB move takes 0.230 ms of NIC, the mover's cap is 87 moves a token, and C2's remote fetch takes 0.26 ms and breaks even at 20 reuses [derived]. With the engine's λ of 37 to 60 µs (`[derived]` in §9.1) the fetch is 0.27 to 0.29 ms and the break-even about 21 to 22 reuses [derived]. | λ is under 70 µs, so C1 and C2 stand. One rail moves 1.22 times one drive's 11.2 GB/s plateau [derived]: the link is still not a faster drive. | two DGX Sparks, one direct cable; the rest in `docs/contrib/dgx-spark.md` §9.1. |
| 6 | `c_node` is 0.553 µs a node (784-node graph, fastest of 3 runs; the 504-node graph gives 0.555 to 0.559). The lookahead's 48 launches cost 0.027 ms a token [derived]. | That is 0.05 to 0.13 % of a 20 to 50 ms step [derived]: nothing that matters on GB10. `docs/research/sparkdec.md` uses this value for `sparkpdl`. | Direct runs of `cnode_probe --time`, not under a lease (a direct run serves, [#13 answer][answer]); built from main `23ee2100` ([#13][measure13]) with `cargo oxide` for sm_86, the driver JIT-compiling the PTX for sm_121; SM clock 2372 to 2424 MHz read after each run (stock). |

## 7. Work in flight (2026-10-11)

The verdicts on the work in flight when the design was written. Items marked landed are in main, and the reasons of the
integrated-arena and §8.1 rows are updated; the rest is as written.

| Item | Verdict | Reason |
|---|---|---|
| R4d-b (seat refusal flip, `paged_set_residency_serves`; landed, `6be94008`) | **keep** | it is the user-facing "residency on a paged plan"; its asserts (landed flips > 0, slot reads > 0, mapping reads 0) are the target's pins, and S8 only tightens the third |
| R4d-c (Paged clause removal + union counters by kind; landed, `967f083f`) | **keep, one change** | the counters-by-kind half is S0's first half — extend its keys to Step/Victim and add `run_ns` in the same Kind change (one `records-refresh`); the clause removal lands as S4 with `GainUnderRuler` as the replacement why, so the unset word on a paged plan carries the derived terms, not "plan a"; r4dc-default.card runs as written |
| refill (landed as S1, `af6d0eba`) | **keep, one condition** | it is S1, the largest single derivable term (−16..−23 ms at the gate plan); its card must run at a shape with fills — at the r4b warm window (fills ≈ 0) the predicted effect is inside the ruler and `lease_take` refuses it; and its measurement is the fault/read split S0 makes readable (`fill_fault_ns`) |
| r7 / r7g (V4.1 and GLM attach) | **keep** | the attach points are §2's owners; two changes after rebase: the per-family expected pick goes through S4's common clause, and `nvtier_family.rs` pins `mapping_reads == 0` as a contract once S2 lands (today printed, not judged) |
| integrated arena (Spark, Q18) | **change** | not a separate plan: it is S7 behind `ArenaMem`, and it follows S1, landed in `af6d0eba` (one give-back site, one time) — the fixed arena priced at load and the Spark ask stand |
| per-part parallel reads (held by the lead) | **absorb into S2** | the part unit is the reader's unit, not a pool-side split; the timeline (pool idle during a decode fill, drive at QD 1) says it pays only on the step path, which S2 moves to the reader anyway |
| r4d-decode.card (R4d-a's A/B) | **keep, re-shape** | its −0.7..0 prediction at the warm window is inside the ruler at 6 rounds in the no-fills regime; run it after S0 at the gate plan where the victim's reader path is exercised beside 45 fills a token, or merge it into S2's card |
| KV pool v2 (landed: `checkpoint/{saved,staging}.rs`, `-c` = the pool) | **keep** | its owners are (l)'s; S12 adds the NVMe tier behind its `NoRoom`; the active sequence's blocks never move (the f × 35–142 ms derivation) |
| PR #16 (the pool invariant, `pool_card_bytes`) | **keep, one doc line** | `pool_card_bytes` is (h)'s budget term for the WC arena; S13 later moves `host`/`unified` into `nodes[0]` (a move) |
| spark2x (two-Spark design, `specs/leader/spark2x/design.md`) | **keep, absorbed** | its Q4 shape is (k); C1–C3 join its S1–S6; its first step (λ on one rail) is the contributor's #5 |
| sparkdec's row-locality prefetch (the contributor's, #17) | **keep** | (h)'s `_sel` gather inherits the per-row `prefetch.global.L2`; (k) #1 measures WC rows under it |
| `docs/contrib/dgx-spark.md` §8.1's decision rule (read in place vs copy) | **close** | #13 measured (ii) WC at 235 = (i) 233 for whole-line reads, (h)'s premise (a random gather keeps that rate only with whole-line loads over 128-B aligned parts, §6.2 row 1, [#13 k-rows][krows]): the band's first branch holds; `dgx-spark.md` §8.1 now states only the branch taken |

## 8. Decisions

Each question the design raised, with its decision. The user decided Q1 to Q12 on 2026-10-11 by their stated
tendencies; the maintainers answered the DGX Spark results on 2026-10-13 ([#13 answer][answer], [#13 devmap][devmap]).

| # | Question | Decision | Date |
|---|---|---|---|
| Q1 | Which shape is the public decode number on the A6000: the 96-token warm window after a prompt (fills ≈ 0, every fill lever 0), or a long decode / the gate plan, where the drive is 55 % of the token? It decides whether S1–S5 are worth opening on the A6000 at all. | both rows go in rig-log. Every fill card is read at a decode with fills (n 1024, or the gate plan). | 2026-10-11 |
| Q2 | The off rule's run-time half: print only (never flip; 10-06 "residency never auto-off"), or a per-window `off` when the measured gain is under the ruler? Recommendation: print only; the plan-time `GainUnderRuler` is the rule. | print only. | 2026-10-11 |
| Q3 | The gate lookahead touches the card graph (one router launch per layer, an id field on the handoff) for a host-tier gain. Open it only on a measured r ≥ 0.6 from the free probe? Recommendation: yes; the probe first, no lease, and the round only if r clears k = m. | the free recall probe first; open it only at r ≥ 0.6. | 2026-10-11 |
| Q4 | `BLOOMERY_HOST_LOCK=1` as the default on a paged plan (the host segment pinned so the page-cache budget is the plan's; needs `RLIMIT_MEMLOCK` or root; a refused lock is recorded, not fatal)? | populate always, lock when permitted, record a refusal. | 2026-10-11 |
| Q5/Q7 | Q5: a pinned arena on discrete cards (`cuMemHostAlloc` for the 20 GB arena: the lane could DMA a promotion straight from the slot without the ring copy; it costs registration time at load and non-swappable RSS)? Recommendation: `auto` = integrated only until S7's clause and a measured registration time exist. Q7: a WC arena on discrete ((i) A′): the lane could DMA a promotion from a `Pinned` slot (−0.3 ms of host thread a promotion, 5.75 a token = −1.7 ms, off the step), but qdot would then walk WC pages, whose CPU rate on 32-B rows is unmeasured (#13 measured a streaming copy only). Recommendation: `Anon` on discrete; one box probe only if the host term ever becomes the step's max. | anonymous memory on a discrete card; `auto` is integrated only. | 2026-10-11 |
| Q6 | The small-host pair (mid vs off on an NVMe-bound token), without which the off rule's (iv) has no ground truth: run it after S2, when the dispatch mutex and the lane's reads are off the chain? Recommendation: yes, after S2, one load an arm. | after S2. | 2026-10-11 |
| Q8 | Lock the KV pool's pageable states (`saved.rs:44-45`) with the host segment under `HOST_LOCK`? Recommendation: no; a sealed state is reclaimable room by (g)'s order, and locking it would take the two-state floor out of the kernel's hands. | no. | 2026-10-11 |
| Q9 | The residency word on GB10: `off` with the why `Unified` (the card mover has nothing to move), or `mid` with 0 flips? Recommendation: `off`/`Unified`; a flip count of 0 would read as a defect. | off, why Unified. | 2026-10-11 |
| Q10 | The N-node home split: static per load (a home change is the mover's), or re-planned per window? Recommendation: static; S4's NIC cap allows ≤ ~~74~~ 87 moves a token at a 40 ms step [derived, [#13 k-rows][krows]], and the plan prints each node's predicted fills (C3). | static per load. | 2026-10-11 |
| Q11 | When do the contributor's six asks ((k), now §6.1) go onto #13? #1, #5 and #6 need no engine code; #2 to #4 want S7 and the `probe_nvread` plain-file arm. Recommendation: post the three now with their decision rules; the rest with S7. | batched through the lead at the next milestone. | 2026-10-11 |
| Q12 | `Kind::State` (S12) before or after 0.3.0? It is the only KV move the derivation admits, and it pays 200× on a 128K resume, but no user has asked for a prompt cache past host RAM. Recommendation: after; keep the derivation. | after 0.3.0. | 2026-10-11 |
| (e) (iii) | `GainUnderRuler` as the plan-time off rule. | GainUnderRuler (iii) is deferred: the step floor is not derivable at plan time. | 2026-10-11 |
| (h) | What a hit costs on GB10 when today's `_sel` gather reads WC pages at a rate between the two branches of §6.1 row 1 (§6.2 row 1). | (h) assumes 128-B aligned parts read with whole-line loads; today's gather rate is recorded beside it; a rate from 160 to 200 GB/s means "record it and reprice (h)". | 2026-10-13 |
| S7 | Is S7's arena rule "≤ 5 s", or a cost priced at load? | The linear cost, priced at load; the 64 GiB time (§6.2 row 2) is a success. | 2026-10-13 |
| S7, card leg | Who builds S7 with its `devmap` clause and the card leg, and on which card are they checked? | A contributor builds both; a design that differs from (d) and (h) stands on its reasons. Any card may check them. The card leg's load shape and part alignment follow measurements. | 2026-10-13 |
| S13 | Does S13's placement half wait for S10? | No; it goes up when ready. | 2026-10-13 |
| §6.2 row 6 | Does a direct run serve for `c_node`, outside a lease? | Yes, with its SM clock and commit stated. | 2026-10-13 |

## 9. Prior work: the designs followed, and where the three diverge

- **Followed:** Mixtral offloading (Eliseev & Mazur, https://arxiv.org/abs/2312.17238: "accelerates offloading by
  taking advantage of innate properties of MoE LLMs") for the LRU cache + gate lookahead pair — with the cache in host
  RAM and the experts *computed on the host* (Fiddler's point, https://arxiv.org/abs/2402.07033: "strategically
  utilizes CPU and GPU resources by determining the optimal execution strategy", 1.26× single batch), not copied to the
  card; and a ranked single reader (MoE-Infinity's "sparsity-aware expert cache, which can trace the sparse activation
  of experts during inference", https://arxiv.org/abs/2401.14361, uses traces for prefetch — ours uses the gate and the
  prompt keep set, no trace at run time). Pre-gated MoE (https://arxiv.org/abs/2308.12066, "pre-gating function which
  alleviates the dynamic nature of sparse expert activation") retrains the router — not applicable to a GGUF we do not
  train.
- **mistral.rs** (HEAD `b0f26d5cd` 2026-09-17, grep by a Haiku sweep): placement is
  fixed at load (`device_map/mappers.rs:63` `set_device` per layer; activations move, `:50-52`), no expert cache, no
  router prefetch; the only runtime weight copies are the diffusion `offloaded` path. Divergence: we move experts
  between three tiers at run time.
- **exllamav3**: no copy on the Mac (`/home/user/exllamav3-src` absent; stated, not read). Its README
  (https://github.com/turboderp-org/exllamav3): "CPU offloading - Allows large MoE models to run with limited GPU
  resources. AVX2 and AVX512 support." — no prefetch or cache named. ik stays the numeric oracle; its `-ot=CPU` is a
  static split. Divergence from all three: a budgeted RAM arena with an NVMe tier behind it and the slot book as the one
  ledger.

[krows]: https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288
[answer]: https://github.com/midagedev/bloomery/issues/13#issuecomment-6134411281
[devmap]: https://github.com/midagedev/bloomery/issues/13#issuecomment-6134437490
[i109]: https://github.com/midagedev/bloomery/issues/109
[lambda]: https://github.com/midagedev/bloomery/pull/44
[measure13]: https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984
[drive13]: https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921
[drive71]: https://github.com/midagedev/bloomery/issues/71#issuecomment-6126758679
