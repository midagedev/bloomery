# tierengine — one tier engine for expert placement across card, RAM and NVMe (design, Fable 5.1, 2026-10-11)

Round `tierengine` (unitier, read-only; no code, no box run, no lease). Tree read: the r4d branch at
WIP `7cec21ad` (R4d-a landed; r4db2's uncommitted edits in `lane.rs`, `record.rs`, `gate_qwen38_serve.rs` not read).
Round `tierengine2` (Fable 5.1, 2026-10-11) added §3 (g)–(l) (host RAM, unified memory, the GPU's NVMe path, read
granularity, N-node Spark, KV + placement), S10–S13 in §5, six rows in §6 and questions 7–12 in §7; the Spark numbers
are the contributor's (#13, #17), the main-tree lines (`workstation.rs`, `ctx.rs`, `checkpoint/`) are read on main.
Tags: [measured, source] a printed number; [derived] arithmetic from code and measured terms; [assumption] stated.
Lines are the r4d tree's unless a path says otherwise. "S" = slots a paged layer carves, "B" = union batch (16),
"fill" = one expert's three parts read into its slot (Qwen3.8: 3.134 MB, slot 3.16 MB).

## 1. The current shape, as built

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

## 2. Per-common-owner attach table

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

`ArenaMem { Anon(AnonMap), Pinned(MappedHost) }` (Q18, one enum at the build, `GpuModel::nvme_tier` `model.rs:1684`
picks by `DeviceInfo::integrated` or the `BLOOMERY_NVTIER_MEM` row). **`give_back` lives in exactly one place and runs
at exactly one time: the release of a slot to the card (`release`, `nvtier.rs:1693`), never before a refill** — the
next fill overwrites the slot's windows, O_DIRECT into mapped pages faults nothing, and the arena's RSS is the plan's
bytes anyway (`HostNeed.arena`). On `Pinned` it is a no-op (the device aliases the physical pages; a `dontneed` would
split the alias — Q18's D1 mutant). The refill round's lever is this deletion. **Page cache budget:** after (b) the only
mapping readers of a paged id are gone (step, prompt, victim, lane stage after excl); the mapping then holds the host
segment + dense tensors, populated at load, and the page cache's size is the plan's host-segment bytes by construction.
The budget is asserted, not managed: `mapping_reads == 0` for paged ids (r4da measured 0, the first anchor), `drops ==
0` once the dropper retires, and a census line `page_cache_bytes` beside `HostNeed` (from `/proc/self/smaps_rollup`'s
file-backed RSS) printed per span. `BLOOMERY_HOST_LOCK=1` on a paged plan keeps the kernel from reclaiming the host
segment under the arena's writes (attach-design Part 2's lower-bound caveat) — a default the user decides (§7).

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
| host segment (`experts`: what the CPU computes) | the paged plan's host picks, populated at load, locked under `HOST_LOCK` (§7 Q4) | 0 — the CPU computes nothing (`docs/contrib/dgx-spark.md` §7: 20 cores beside 48 SMs on one 273 GB/s add contention) | fixed | placement (`expert_nvme_tier`'s H/2 rule, `placement.rs:2864`) |
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
the box). GB10: 125.2 − 111.3 = 13.9 GB for KV and runtime [derived, dgx-spark.md:110] → at 128K one slot's state 3.98
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
   (a slot the card reads in place at 235 GB/s = `cuMemAlloc`'s 233: a copy between them saves 0 bytes a use and costs
   2 × 3.134 MB of pool traffic a fill); `File` stays. Card residency on GB10 *is* arena residency: a layer's "card
   experts" are the slots of the one arena, S_pool = ⌊(pool − dense − KV − floor) / slot bytes / layers⌋; the card
   mover is `off` with the why `Unified` (no second level to move to); the book's heat ranks eviction (arena → file)
   and the lookahead's prefetch (file → arena) only. Promotion is the fill itself. The step port's leg is the card leg
   (`TierLeg::Card`): `TierPin` yields `device_slot(layer, id, part) -> u64` (Q18) from `cuMemHostGetDevicePointer` on
   the arena's base, and the card's `_sel` gather per quant family (the `q4k_gemv_sel` shape) runs the pinned batch from
   those addresses — the one kernel-side addition, per quant, never per model.
2. **The arena is `ArenaMem::Pinned`: one `cuMemHostAlloc(PORTABLE|DEVICEMAP|WRITECOMBINED)` per `LayerArena`**,
   Q18's plan. The `Registered` variant goes (157–160 GB/s registered or not [measured, #13]: worth 0). On discrete the
   arena stays `Anon`: the CPU reads it 480 picks a token, WC "cannot be read efficiently by most CPUs", and the
   27.8 GB/s streaming figure says nothing about qdot's 32-B row walks. `give_back` is a no-op on `Pinned` ((d)): the
   arena is a fixed pool allocation priced at load, counted in `HostNeed.cards` / `pool_card_bytes` (`placement.rs:859`),
   and a `MADV_DONTNEED` would only split the device alias (D1). A fill overwrites the slot's windows; the DMA write of
   3.134 MB into the pool is 13 µs of the 235 GB/s [derived], paid once, no copy after it.
3. **The pool's budget is one number.** #16's `PoolOver` (Σ card + host ≤ `MemAvailable`) stays the refusal; the
   arena's `cards` term makes the (g) table's "arena" and "card experts" rows one row on GB10. A model above the pool
   (GLM UD-Q4_K_XL 199.7 GB, V4.1 Q3_K_M 347 GB) is the only reason the NVMe tier exists there; a model inside it
   (Qwen3.8 UD-Q4_K_XL 111.3 GB) plans every expert into `Pool`: S = all, 0 fills.
4. **Per fill on GB10** [derived from #13's drive]: 3 parts of ~1.045 MB at QD 3 ≈ 7.3 GB/s (Q1 3.73 → Q4 8.29,
   interpolated) = 0.43 ms; 0.28 ms at the 11.2 GB/s plateau; the CPU's part is three preads (≈ 30 µs [assumption,
   10 µs a pread]). A hit costs the card 3.134 MB / 235 GB/s = 13 µs an expert: the host leg's 23.6 µs and the whole
   host-CPU row of (c)'s timeline are gone on GB10. Above the pool the decode wall is card bytes at 235 + fills ×
   0.28–0.43 ms: GLM on one Spark, 88 GB paged [spark2x capacity row], at the box's gate-plan miss rate (45.7 fills a
   token [measured]) adds 13–20 ms to a ~50 ms step [derived; the Spark's own miss rate is (k)'s ask #4].

Still unmeasured after #13: a `_sel`-shaped gather (random slot rows, 32-B strided) from WC pages against the
sequential `hostread2` rows kernel, and the time of one ≥ 60 GB `cuMemHostAlloc` ((k)'s #1 and #2).

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
| A′. as A, `Pinned` arena (§7 Q5): the lane DMAs from the slot | 3.134 | write 3.134; the DMA reads 3.16 | 0 | 30 µs | 0.64 + 0.12 |
| B. GDS/cuFile direct, A6000 only: drive → VRAM | 3.134 | 0 | 0 | the cuFile call | 0.64–1.25 ms, paid per card use |
| C. GB10 in-place: `O_DIRECT` → WC arena; the SMs read it | 3.134 | write 3.134 | 0 | 30 µs | 0.28–0.43 ms; nothing after |
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
| unit | the part (3 a fill), as (b) and S2 | 1.045 MB = 1 command; a fill at QD 3 ≈ 4.9 GB/s (4.15 → 5.67 interpolated) = 0.64 ms vs 1.25 serial [derived] | 1 MiB = 8 commands; QD 3 ≈ 7.3 GB/s = 0.43 ms [derived] |
| sub-unit (S2b: only if S2's measured QD-3 fill is ≥ 0.6 ms) | part / 3 ≈ 350 KB, so one fill carries 9 units ≥ the knee | QD 9 ≈ 6.5 GB/s: 0.48 ms, −0.16 ms a fill = −7 ms a token at 45.7 fills (−8 % of 85) [derived]; +6 preads ≈ +60 µs CPU across threads; waste 2.3 % | ≈ Q9 ~10 GB/s [interpolated Q4 8.29 → Q16 11.16]: 0.31 ms, −0.12 a fill |
| queue depth (reader threads) | the plateau's QD | 16 (6.77; QD 8 is 96 % of it) | 16 (11.16; Q32 11.20) |
| background share during steps | Victim + Prefetch ≤ the knee, so a Step's 3 units take threads at once | ≤ 8 threads: a Step unit never waits a background unit's sojourn (16.7 MB / 6.77 GB/s = 2.47 ms at a full QD 16) | ≤ 8 |
| prefetch depth, decode | one layer ahead: a layer's compute (37 ms / 48 = 0.77 ms at the gate plan) ≥ a QD-3 fill (0.64) hides it; two ahead hides nothing more and doubles the (1 − r) waste [derived] | 1 layer, ≤ 3 × misses units (≈ 3 at 0.95 miss a layer) | 1 layer |
| prefetch depth, prompt | one batch ahead (built): 16 picks = 48 units = QD 16 | 48 units | 48 units |
| bytes in flight (Little: L = N / R) | at the plateau | 16 × 1.045 = 16.7 MB; a unit's sojourn 2.47 ms (the prompt's regime); an idle-queue Step fill sees QD 3: 0.64 ms | 16.8 MB; 1.5 ms; Step 0.43 ms |

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
§8.1): the NIC reaches only `cudaHostAlloc` pages — which the WC arena *is*. One cable measured 189.85 Gb/s = 23.7 GB/s
a direction, one rail and QP 92.57 Gb/s = 11.6 GB/s (NVIDIA's playbook, §8.1). λ (one round trip) is unmeasured;
spark2x derives 25–70 µs.

Decision — **a per-node engine over a static placement split** (spark2x Q4's shape), by three numeric criteria:
- **C1, claims are local.** Books per node, `(node, layer, id) → state`, each node its own `NvTier` over its own pool
  arena, drive and file copy. The alternative, one global book on node 0 with remote claims, puts λ on every claim: 480
  claims a token × 2 messages × 25–70 µs = 24–67 ms against a 20–50 ms step → refused for every N ≥ 2. The rule: λ ×
  remote claims a token ≤ 1 % of the step (≤ 3–12 claims a token), which only a local book meets.
- **C2, no remote-arena tier.** The I/O scheduler is per node: a node's misses go to its own drive (a 4 TB Gen5 part
  each, 11.2 GB/s plateau each — drives scale with N, rails do not). Node j's pool as node 0's slow tier (a fetch of
  3.134 MB a use at 11.6 GB/s + λ = 0.30–0.34 ms) pays only when the id is reused ≥ 0.30 / 0.013 ≈ 23 times before
  node 0 evicts it, since node j computes it in place in 13 µs + λ — a hot id the split already homes where it is
  used. The NVMe plateau (11.2) equals one rail (11.6): the link is not a faster drive.
- **C3, balance includes fills.** The window per layer is the max over nodes of (its slots × 13 µs + its misses ×
  0.28–0.43 ms + λ for a remote); a node with the colder set bounds the step, so the balancer's objective gains the
  predicted fills per node within 10 % of each other (spark2x's S2-ii extended to fills) beside S3's byte balance.
- Heat stays one source: node 0's tally ships E × L u32 = 512 × 48 × 4 B = 98 KB a window [derived] to each node for
  its eviction; the cross-node mover (Pool@i ↔ Pool@j, 3.134 MB a move at 11.6 GB/s = 0.27 ms of NIC, off the step)
  rebalances on heat drift under S4 (NIC ≤ 50 % of the step: ≤ 74 moves a token at a 40 ms step — never binding). The
  split is static per load (each paged id has a home node from the byte ledger, S2-i per node); the runtime moves only
  inside a node's budgets; a home change is the mover's, never the plan's.

The contributor's part — the Spark-only facts the design cannot derive, one command each:

| # | Fact | Command (one) | The number decides | We build first |
|---|---|---|---|---|
| 1 | a `_sel`-shaped gather from WC pages: random slot rows, 32-B strided, vs `hostread2`'s sequential rows | `hostread2 2 host-wc` with a `ROWS=random` arm (≈ 20 lines on the comment's kernel; we post the patch on #13) | ≥ 200 GB/s → (h) as written; < 160 → each slot row gets a 128-B pad (a plan-bytes change) | nothing: the kernel is the contributor's |
| 2 | one `cuMemHostAlloc(WC)` of 64 GiB: success and seconds | `FILL=$M hostread2 1 host-wc` with a 64 GiB size arg | ≤ 5 s → S7 prices the arena at load; failure → S7 allocates per layer (48 calls) | S7 (`ArenaMem::Pinned`, `device_slot`) |
| 3 | the part-unit curve on the Spark's drive at our shape: 1.045 MB, QD 3 / 8 / 16, random offsets | `probe_nvread --threads 3,8,16 --file $M --unit 1045000` | QD-3 ≥ 7 GB/s → (j)'s unit holds; < 6 → S2b's tiles on the Spark | a plain-file arm of `probe_nvread` (today it opens a qwen4exp split set, #13) |
| 4 | fills a token above the pool: GLM UD-Q4_K_XL on one Spark, 88 GB paged | `bloomery-serve … --place auto`, S0's `nvtier` record, one 512 prompt + 96 steps | fills × 0.43 ms against the step: > 20 % → (c)'s lookahead opens on GB10 first | S0 (records by kind), S7 |
| 5 | λ and the 3 MiB-message rate on one rail | `ib_write_lat` / `ib_write_bw -s 3145728` between two Sparks (spark2x Q5's first step) | λ ≤ 70 µs → C1/C2 as derived; the move rate sets the mover's cap | nothing |
| 6 | GB10 `c_node` | `cnode_probe --time` (sparkdec §9) | the lookahead's 48 launches a token (0.04 ms on the A6000) on GB10 | nothing: the bin exists |

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

### Design followed, and where the three diverge

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

| # | Step | Class, proof | Gates, FAIL-first | The one box measurement, expected | Deletes |
|---|---|---|---|---|---|
| S0 | **Records by kind + the runner keeps arm stderr** (f): `KindStats.run_ns`, `{step,keep,transient}_*`, `fill_{fault,read}_ns`; `depth-qwen3moe.sh`/`ab-decode.sh` write `<log>.arm<n>.err` and fold `nvtier`/`residency` into the row | records (Kind change → `records-refresh`); host only, ptx-scan identical | `gate-gpu-nvtier` kind test (re-pinned with its derivation); runner self-test: an arm without stderr → red | none (a functional gate run prints the keys; the fault/read split is read on the next card as a beside field) | `count_fill`'s `pinned` key; the arm-record gap named in r4d-decode.card and r4dc-default.card |
| S1 | **refill**: delete the pre-read `give_back` in `read_pick` (`:2243`, `PickRead.give_back`, `Pick.gone`'s use); keep `release`'s | host dispatch path → same-lease A/B (refill's card, at the gate plan or n ≥ 1024) | `gate-gpu-nvtier` bit identity (same bytes, the audit); FAIL-first: a short read must still refuse by name (`:2252`) | fill 1.05 → 0.6–0.75 ms, −16..−23 ms a token at the gate plan (+19..+37 % tok/s); at the r4b warm window inside the ruler → refused | the refault; `AnonMap::dontneed` keeps one caller |
| S2 | **One reader, part units, `Kind::Step`** (b): the step's misses submitted to the reader at rank 0, the chain waits on the latch; unit = part; Victim/Prefetch take a thread only while Step is empty | host dispatch path; ptx-scan identical; the reader's count identity `fills == step_units/3 + prompt_units/3 + victim_units/3` | `gate-gpu-nvtier`: a new clause "the pool makes no NVMe read" (`step_jobs > 0`, pool fill count 0); FAIL-first: route Step through the pool → red; a Victim ahead of a queued Step → red | step fill 0.63–0.75 → net with S1 ≥ 0.46 ms; −14..−19 ms a token at the gate plan on top of S1 (bounded: S1+S2 ≤ −27 ms); the hand-off 0.05–0.15 ms a job is the term to watch (noninf margin 5 %) | `NvTier::fill` (pool path), `ensure`/`ensure_as` sync forms, `fill_victims`' own submit (a victim is a job like any), the lane's blocking under the receive lock while the reader runs (the lane submits, then stages when the latch opens) |
| S3 | **In-layer overlap**: `experts_into_on` computes the hits while the Step job is in flight; per-expert outputs stored, summed in list order | host path; bits identical by construction (same order of the final sum); proof: `gate-gpu-e2e` tokens equal | `gate-gpu-nvtier` steps clause; FAIL-first: sum in landing order → the bit pin red | −5..−10 ms a token at the gate plan (0.21 ms × layers with a miss); 0 at the warm window | nothing (the fill's join moves after the hits' compute) |
| S4 | **Off rule derived** (e): `GainUnderRuler` in `room_for`, the `Paged` clause's 36 sites go (r4dc2), one `RoomShort` → each family's why | host plan path; levers unit tests; proof: the probe table of r4dc1 §2 reproduces (plan a → `mid-p0-s1`, GLM 3090 → `off` NoCardExperts) | levers tests (FAIL-first: restore the clause → the paged-shape pin red); `gate_nvtier` clause 5 re-pinned to the word; `nvtier_family` per-family pick | r4dc-default.card as written (−2..+10 %, 10 rounds, the flip's own A/B) | `PlanTier`, `PagedText`, `ResidencyWhy::Paged`, `Residency38Why::Paged`, `GlmWhy::Paged`, `RoomShort::Paged` |
| S5 | **Gate lookahead** (c): probe first (a `lookahead` record, functional, no lease); then the round: router_{l+1} on the pre-attention residual in the card graph, k ids on the handoff page, `Kind::Prefetch` jobs cancelled by a landing Step claim | add class on the card (ptx-scan = base + the router entry, `gate-ptx-spill` red on it before pinning) + host path; launch count Δt = 48 × c_node | `gate-gpu-e2e` (bits: the real router unchanged); new nvtier clause `prefetch_hit + cancelled + wasted == issued`; FAIL-first: a prefetch that claims a pinned slot → red | the probe: r at k = m, expected 0.6–0.85; the round's A/B at the gate plan: −10..−20 ms a token at r 0.6–0.85 after S1–S3; refused below r 0.5 | nothing until measured; at r < 0.5 the step is dropped and (c) closes with the three slot-level levers |
| S6 | **Heat in the book** (a): one count per (layer, id) written by the tally, read by the rule and `pick_slot`; `SwapRule.counts` a view | move; proof: a Mac route-trace replay plans identical flips from both sources; `ptx-scan` identical | `gate-gpu-qwen38-residency` / `gate-gpu-swap` flips pin; `gate-gpu-nvtier` eviction clause (coldest count, transient first); FAIL-first: evict by tick alone → the clause red | none (counting identity); read beside: fills a token at n 1024 (expected to fall with count-ranked eviction vs tick LRU, unmeasured) | `SwapRule.counts` as a second copy; `NvTier::Heat`/`slot_heat` (no caller) or they become the owner |
| S7 | **`ArenaMem::Pinned`** (d): the enum, `give_back` no-op on Pinned, `device_slot` for an integrated read, `devmap` clause on the A6000 | add behind `BLOOMERY_NVTIER_MEM` (default `auto` = integrated only) | Q18's clause: DtoH of `device_slot` == `slot()` bytes; FAIL-first D1 (dontneed on Pinned → stale alias red), D2 (part skew dropped → red) | the one term: `cuMemHostRegister`/`cuMemHostAlloc` over 20 GB works and its time (expected 2–4 s [assumption]); bits equal | the integrated arena as a separate plan |
| S8 | **Dropper and mapping path for paged ids go** | delete; per entry the grep with no caller; `mapping_reads == 0` and `drops == 0` pinned | `gate-gpu-nvtier` clause 9 `exclusive` turns the tolerated mapping reads into a 0 pin (dated); FAIL-first: a lane stage of an unfilled id through the mapping → red | none (a functional run prints the pins) | `nvtier-drop`, `drop_layer`, `end_read`'s queue, `PageDrop` for paged ids, `source()`'s mapping fallthrough for arena ids |
| S9 | **Lift the copies** (§2): Qwen3.8's host-run build onto `model.rs:1710`; one `placement::Room` read per `describe`; MiMo's body attach through the same call | move; `ptx-scan` identical; `just affected --narrow` | each family's nvtier gate (`nvtier_family.rs` with a MiMo caller) | none | `body38.rs:1451-1456`'s own build; three room fields |
| S10 | **One budget solver** (g, l): a `MemoryModel` per device (`Card | Host | Pool`, capacity, bandwidth, reclaim class) whose terms are card experts / S_pool, the KV pool at ctx × slots × state bytes, the arena's S, the cache budget and floor, the churn pool, the page-cache assertion; the (g) reclaim order at a serve-time `--ctx` | host plan path, pure crate; proof: every plan in `tools/ref/plan-room.sh`'s room prints the same budgets as before (a counting identity) and the four replaced rules' tests re-point to the solver's terms | placement lib tests; `gate-gpu-qwen38-serve`'s two paged clauses re-pinned to the terms (dated); FAIL-first: a reclaim that touches the host segment → red; a cache under the two-state floor → red | none | `expert_nvme_tier`'s room arithmetic, `ArenaFloor` as a free rule, `grow_kv`, the paged cache default, the per-family room fields (with S9) |
| S11 | **Reader shares and per-drive parameters** (j): background ≤ 8 of 16 threads during steps; `max_sectors_kb`, r1, R_p, the knee read at load; S2b (part/3 tiles) opened only if S2's measured QD-3 fill ≥ 0.6 ms | host path; the reader's unit tests | `gate-gpu-nvtier` clause: a Step unit's `wait_ns` ≤ one background unit's service time; FAIL-first: a 16-thread background class → red | S2b's own card: −0.16 ms a fill, −7 ms a token at the gate plan (−8 %) | nothing |
| S12 | **`Kind::State`** (l): sealed states' NVMe tier behind the pool's `NoRoom`, through the reader at rank 4 | add, host only; `ptx-scan` identical | `gate-gpu-qwen38-serve` clause: a state evicted to NVMe resumes bit-identical (tokens equal); FAIL-first: one page short on the restore → red | a 32K restore 0.18 s (1.19 GB / 6.77 GB/s), functional | the request-path fallback for a sealed state that found no room |
| S13 | **N-node** (k): `Machine.nodes` + `Fabric` (spark2x Amendment 1), per-node `NvTier`, the 98 KB heat window, the mover lane under S4's cap | placement first (pure; `nodes_scale` + C3), then the lane; after S7 and the contributor's #5 | `nodes_scale` with the per-node fills criterion; the lane's bit identity on two processes of one box (loopback) | λ on one rail (the contributor's) | nothing in this tree |

Order and trains: S0 → S1 (in flight) → S2 → S3 → S4 + S10 (one owner of `placement`/`levers`: S10 replaces the same
rules r4dc2's clause removal touches) → S5 probe (free, any time after S0) → S6 → S7 → S11 (reads S2's measured QD-3
fill) → S8 → S9 → S12 → S13. S1+S2 share `read_pick`/`fill`: S2 rebases on S1. S2 and S4 touch `gate_nvtier.rs` and
`nvtier.rs`: one owner per round, S4 after S2. Every step deletes or converts a path within two steps: S2 removes the
pool reader that S1 touched; S8 removes the mapping path S2 made unreachable; S10 removes the four scattered budget
rules the moment the solver prints their numbers. GB10-only behaviour enters as parameters of S7, S10 and S11
(`ArenaMem::Pinned`, the `Pool` device kind, the drive's constants), never as a step of its own.

## 6. The in-flight work

| Item | Verdict | Reason |
|---|---|---|
| R4d-b (seat refusal flip, `paged_set_residency_serves`) | **keep** | it is the user-facing "residency on a paged plan"; its asserts (landed flips > 0, slot reads > 0, mapping reads 0) are the target's pins, and S8 only tightens the third |
| R4d-c (Paged clause removal + union counters by kind) | **keep, one change** | the counters-by-kind half is S0's first half — extend its keys to Step/Victim and add `run_ns` in the same Kind change (one `records-refresh`); the clause removal lands as S4 with `GainUnderRuler` as the replacement why, so the unset word on a paged plan carries the derived terms, not "plan a"; r4dc-default.card runs as written |
| refill | **keep, one condition** | it is S1, the largest single derivable term (−16..−23 ms at the gate plan); its card must run at a shape with fills — at the r4b warm window (fills ≈ 0) the predicted effect is inside the ruler and `lease_take` refuses it; and its measurement is the fault/read split S0 makes readable (`fill_fault_ns`) |
| r7 / r7g (V4.1 and GLM attach) | **keep** | the attach points are §2's owners; two changes after rebase: the per-family expected pick goes through S4's common clause, and `nvtier_family.rs` pins `mapping_reads == 0` as a contract once S2 lands (today printed, not judged) |
| integrated arena (Spark, Q18) | **change** | not a separate plan: it is S7 behind `ArenaMem`, and it waits for S1 (one give-back site, one time) — the fixed arena priced at load and the Spark ask stand |
| per-part parallel reads (held by the lead) | **absorb into S2** | the part unit is the reader's unit, not a pool-side split; the timeline (pool idle during a decode fill, drive at QD 1) says it pays only on the step path, which S2 moves to the reader anyway |
| r4d-decode.card (R4d-a's A/B) | **keep, re-shape** | its −0.7..0 prediction at the warm window is inside the ruler at 6 rounds in the no-fills regime; run it after S0 at the gate plan where the victim's reader path is exercised beside 45 fills a token, or merge it into S2's card |
| KV pool v2 (landed: `checkpoint/{saved,staging}.rs`, `-c` = the pool) | **keep** | its owners are (l)'s; S12 adds the NVMe tier behind its `NoRoom`; the active sequence's blocks never move (the f × 35–142 ms derivation) |
| PR #16 (the pool invariant, `pool_card_bytes`) | **keep, one doc line** | `pool_card_bytes` is (h)'s budget term for the WC arena; S13 later moves `host`/`unified` into `nodes[0]` (a move) |
| spark2x (two-Spark design, `specs/leader/spark2x/design.md`) | **keep, absorbed** | its Q4 shape is (k); C1–C3 join its S1–S6; its first step (λ on one rail) is the contributor's #5 |
| sparkdec's row-locality prefetch (the contributor's, #17) | **keep** | (h)'s `_sel` gather inherits the per-row `prefetch.global.L2`; (k) #1 measures WC rows under it |
| `docs/contrib/dgx-spark.md` §7's decision rule (read in place vs copy) | **close** | #13 measured (ii) WC at 235 = (i) 233: the band's first branch holds, so the rule's three fallbacks (hugepage, copy on the copy engines) are dead text to remove with S7 |

## 7. Open questions for the user

1. **Which shape is the public decode number on the A6000 — the 96-token warm window after a prompt (fills ≈ 0, every
   fill lever 0) or a long decode / the gate plan where the drive is 55 % of the token?** It decides whether S1–S5 are
   worth opening on the A6000 at all. Recommendation: both rows in rig-log; every fill card at the gate plan or n 1024.
2. **The off rule's run-time half**: print only (never flip; 10-06 "residency never auto-off"), or allow a per-window
   `off` when the measured gain is under the ruler? Recommendation: print only; the plan-time `GainUnderRuler` is the
   rule.
3. **The gate lookahead touches the card graph** (one router launch per layer, an id field on the handoff) for a
   host-tier gain. Open it only on a measured r ≥ 0.6 from the free probe? Recommendation: yes — the probe first, no
   lease, and the round only if r clears k = m.
4. **`BLOOMERY_HOST_LOCK=1` as the default on a paged plan** (the host segment pinned so the page-cache budget is the
   plan's; needs `RLIMIT_MEMLOCK` or root; a refused lock is recorded, not fatal)? Recommendation: populate always, lock
   when permitted, record the refusal.
5. **A pinned arena on discrete cards** (`cuMemHostAlloc` for the 20 GB arena: the lane could DMA a promotion straight
   from the slot without the ring copy; costs registration time at load and non-swappable RSS). Recommendation: `auto` =
   integrated only until S7's clause and a measured registration time exist.
6. **The small-host pair** (mid vs off on an NVMe-bound token) — the off rule's (iv) has no ground truth without
   it. Run it after S2 (the dispatch mutex and the lane's reads are then off the chain)? Recommendation: yes, after S2,
   one load an arm.
7. **A WC arena on discrete** (Q5 revisited by (i) A′): the lane could DMA a promotion from a `Pinned` slot (−0.3 ms of
   host thread a promotion, 5.75 a token = −1.7 ms, off the step), but qdot would then walk WC pages, whose CPU rate on
   32-B rows is unmeasured (#13 measured a streaming copy only). Recommendation: `Anon` on discrete; one box probe only
   if the host term ever becomes the step's max.
8. **The KV pool's pageable states under `HOST_LOCK`** (today pageable by design, `saved.rs:44-45`): lock them with the
   host segment? Recommendation: no — a sealed state is reclaimable room by (g)'s order, and locking it would take the
   two-state floor out of the kernel's hands.
9. **The residency word on GB10**: print `off` with the why `Unified` (the card mover has nothing to move), or `mid`
   with 0 flips? Recommendation: `off`/`Unified`; a flip count of 0 would read as a defect.
10. **The N-node home split**: static per load (a home change is the mover's) or re-planned per window?
    Recommendation: static; S4's NIC cap allows ≤ 74 moves a token at a 40 ms step, and the plan prints each node's
    predicted fills (C3).
11. **When the contributor's six asks go onto #13**: #1, #5, #6 need no engine code and can go now; #2–#4 want S7 and
    the `probe_nvread` plain-file arm. Recommendation: post the three now with their decision rules; the rest with S7.
12. **`Kind::State` (S12) before or after 0.3.0?** It is the only KV move the derivation admits, and it pays 200× on a
    128K resume, but no user has asked for a prompt cache past host RAM. Recommendation: after; keep the derivation.

## 8. Decided (2026-10-11)

The user's rule: decide by their stated tendencies. Each answer below closes the matching §7 question.
- Q1: both rows go in rig-log. Every fill card is read at a decode with fills (n 1024, or the gate plan).
- Q2 (the off rule's run-time half): print only.
- Q3 (lookahead): the free recall probe first; open it only at r ≥ 0.6.
- Q4 (HOST_LOCK on a paged plan): populate always, lock when permitted, record a refusal.
- Q5/Q7 (pinned or write-combined arena on a discrete card): anonymous memory on a discrete card; `auto` is integrated only.
- Q6 (the small-host pair): after S2.
- Q8 (lock the KV states): no.
- Q9 (the GB10 word): off, why Unified.
- Q10 (the N-node home): static per load.
- Q11 (contributor asks): batched through the lead at the next milestone.
- Q12 (Kind::State): after 0.3.0.
- GainUnderRuler (iii) is deferred: the step floor is not derivable at plan time.
