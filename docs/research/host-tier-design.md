# HostTier: one host tier, two ports (round `hosttier`)

**Status.** Design memo from the read-only round `hosttier` (2026-09-27). 03 designs `HostTier` and aa signs it
before any code (`docs/rebuild.md:341`). **aa signed §7 items 1–11 and 13 on 2026-09-27, and item 12 with a
condition** (the end of §7; the conditions also set the order of step B against act-planes' B, and what hostcfg
leaves to `gpumodel`). The implementation is §6: `hostcfg` and `benchprune` first, then `hostone` (steps A–E). The
memo was read before r8land landed; §6.4 lists the seven places r8land can change an answer.

## Provenance

**Main tree.** Read at `6bc4db6`. Every `.rs` file cited here is byte-identical at `8e4b4f7`: `git diff --stat 6bc4db6 8e4b4f7` touches no `crates/*/src` file.

**r8host.** Read at `0e19002` (branch `r8host`) for its row-unit layout. r8land has not landed; §6.4 lists where it can change an answer.

**Reference trees on the box:**
- ik_llama.cpp `c10fbbcc`.
- mistral.rs `d5ae0f1`, plus an uncommitted 5-line edit in `mistralrs-quant/src/gguf/cpu.rs:80-83`. The edit is in the very function §5 cites and changes only the fallback's dtype; which path runs is unchanged.
- exllamav3 `0740edc`, plus one untracked test file.

**GLM-5.3-Flash UD-Q4_K_XL** is not on the box. Its routed types come from the lead's header read (`docs/research/modelspec-design.md:9-15`). Its shapes come from the UD-Q2_K_XL header (`:497-536`).

**Method.** Nothing was built or run. Every number is either measured, with its source, or marked [derived].

## 0. Decomposition, in brief

Every term of the question is one of two kinds:
- a count read from the code (§2.1);
- arithmetic on numbers already measured (§4).

Two terms could have needed a box run: claim contention and the steal-block cap. The design removes them as differences instead of pricing them (§2.2). So no term needs a run, and this round wrote no card.

## 1. The types

### 1.1 Today

| Fact | Owner 1 | Owner 2 |
|---|---|---|
| Fault read | `Gpu::fault`, `lib.rs:2240-2278`: an async copy plus an engine-stream sync, once per word | `read_fault`, `hybrid.rs:2326-2355`: a synchronous `cuMemcpyDtoH` that waits on no stream |
| Mapped host memory | `MappedHost`, `hybrid.rs:464-594`: `DEVICEMAP` only, `Send` | `HostFlags`, `graph.rs:328-464`: `PORTABLE\|DEVICEMAP`, `Send + Sync` |
| Memop builders | `op_*` and `mem_batch`, `hybrid.rs:599-676` | built inline in `HostFlags::enqueue_wait`, `graph.rs:431-454` |
| `capturing` | `hybrid.rs:679-685`, private | `graph.rs:300-307`, public; called at `body.rs:1405` and `prefill.rs:1216` |
| Batch state machine | `batch.rs:964-1164` (`mod exchange`, gpu-deepseek41) | its entry, poison and stats sit in `Hybrid` (`hybrid.rs:1796-1946`, `1150-1210`) |

`Hybrid<H>` (`hybrid.rs:1407-1454`) is the tier and both ports at once. It holds:
- the step's words: `served`, `captured`, `chain`, `capturing`, `step_refusal`, `pair_row0`;
- the batch's scratch: `batch_lists`, `batch_lens`, `batch_refused`, `batch_slices`;
- the health they share: `poisoned`, `poison`, `failing`, `refusal`;
- an optional fault word;
- one `HybridStats` covering all of it.

Two more things sit in odd places:
- The slot map lives in the step's `Boundary` (`slots`, `hybrid.rs:752`), yet the batch reads it too.
- The host set, `HostResidency`, lives in `GpuModel` (`model.rs:352`).

### 1.2 The design

```rust
// crates/gpu/src/host/{mod,step,batch,slots}.rs — hybrid.rs split by owner

/// The model's host computation: one call for every port.
pub trait HostExperts {
    /// Column j of `x` with its host list `lists[j]`, summed into
    /// `out[j·hidden..]`, through `scratch` (the calling port's): bit for bit
    /// per column whatever the width.
    fn experts(&mut self, layer: usize, x: Tensor2View<'_>, lists: &[&[(u32, f32)]],
               out: &mut [f32], scratch: &mut UnionScratch) -> Result<(), GpuError>;
}

pub struct HostTier<E: HostExperts> {
    residency: Option<HostResidency>, // option A: from GpuModel (model.rs:352); declared before
                                      // `experts` — the lock's spans are pages of the mapping
                                      // `experts` keeps alive. Option B: stays in GpuModel (§7 item 6).
    experts: E,                       // Ds41Host (ffn.rs:1502), PlanHost (deepseek2/mod.rs:86-132), stubs
    slots: SlotMap,                   // from Boundary (hybrid.rs:752)
    fault: Arc<DeviceBuffer<u32>>,    // a constructor argument: no unwatched tier
    health: Health,                   // poisoned, poison, failing, refusal (hybrid.rs:1425-1436, 1449)
    stats: TierStats,                 // refusals, resets, last_poison
    pub step: StepPort,
    batch: Option<BatchPort>,         // made at the first prompt or by prepare, as Ds41Host::prepare_union is
}

pub struct StepPort {
    boundary: Boundary,       // region, windows, page, rows (slots moved out)
    served: u32, captured: [Vec<(usize, usize)>; 2], chain: Chain, capturing: bool,
    step_refusal: Option<Refusal>, pair_row0: Option<PairRow0>,
    narrow: UnionScratch,     // m ≤ 8 columns, made at load
    stats: StepStats,         // served … pair_row1_slots (hybrid.rs:1153-1190)
}

pub struct BatchPort {
    sets: [Set; 2],           // batch.rs:984-991, moved
    n_embd: usize, n_used: usize, cap: usize, down: usize, serve: usize, up: usize,
    lists: Vec<(u32, f32)>, lens: Vec<usize>, refused: Vec<usize>, slices: Vec<&'static [(u32, f32)]>,
    wide: UnionScratch,       // UNION_MAX_COLS columns (moe.rs:960)
    stats: BatchStats,        // batch_served, batch_cols, batch_host_slots, batch_ns
}
```

**Constructors.** Both run at load time only.
- `HostTier::new(experts, slots, residency, fault, step, cfg: HostCfg)` checks the word's context through `module_fault_word`, as `watch_fault` does today (`hybrid.rs:1495-1499`).
- `StepPort::new(ctx, stream, PageShape { hidden, n_used, units, cols }, overlap)`.

**Holder.** The body holds the tier, as it holds `Hybrid` today (`body.rs:2258-2259`).
- `ChainBody::serve_replay` and `serve_replay_pair` (`body.rs:2123`, `2154`) call `tier.serve_captured(chain)`.
- The session design's `HostServed` capability (`docs/research/session-design.md:225`, `:539`) is that call: the body's host service. It is not `HostExperts`, which is the model's compute.

**Records.** `tier.stats()` returns a view with `HybridStats`' fields, so these readers see the same values until a schema round moves them: `record.rs:850-901`, `generate_ds41.rs:1016-1019, 1607-1728`, `prefill.rs:1296-1315`.

### 1.3 StepPort protocol (per go; captured)

| # | State and transition | Who | Waits on |
|---|---|---|---|
| 1 | The image is written. For V4.1 the handoff kernel writes the row's image into the page (`ffn.rs:1266`); for other models a copy node does it (`enqueue_out`, `hybrid.rs:1022-1039`). | card | the stream: the norm and router are done (`ffn.rs:1229-1246`) |
| 2 | Go: `[barrier_sys, write Lyr(r)=layer, barrier_sys, Gen += 1, SEQ += 1]` (`go_batch`, `hybrid.rs:1064-1076`). | card | the PCIe flush (the system barrier) |
| 3 | `wait_go` (called at `hybrid.rs:2020`, defined at `2187-2218`). Every pool thread spins on `Gen ≥ want`. Every 4,096 spins it polls the 10 s deadline and the launch-failure flag. This is itself a pool dispatch. | host, 32 threads | the mapped write of `Gen` |
| 4 | Check: the generation is not `in_flight` or more ahead; the sequence equals `served`; `Lyr(r)` is the layer (`hybrid.rs:2030-2070`). | host, calling thread | nothing |
| 5 | Serve: build the list from the slot map (`hybrid.rs:2071-2097`), copy x (`:2099`), run `non_finite` (`:2109-2112`), run the experts (`:2138`). | host pool | DRAM (the weights) |
| 6 | Signal: `Cnt(r) += 1` with Release (`hybrid.rs:2139-2142`); then `served = want`. | host | nothing |
| 7 | Card wait: `[wait_geq Cnt(r) ≥ 1, Cnt(r) −= 1]` (`hybrid.rs:1085-1093`). With overlap on it is enqueued at `ffn.rs:1095`, with overlap off at `ffn.rs:1270`. Then the join reads `hsum` from the page. | card | the host's write, over PCIe |
| F | Failure: poison the tier. Then `release()` sets every `Cnt` to `RELEASE = 1 << 30` (`hybrid.rs:1097-1103`, `455`), so every pending wait passes and the stream drains. After the drain the step's caller names the refusal: `take_step_refusal` (`:1513`), then `name_refusal` (`:2298`). | host | nothing |

**Invariant needed to read x in place (§2.1, row 10).** Row r's image does not change between its go (step 2) and the host raising `Cnt(r)` (step 6). The reason is stream order:
- Row r's next write to its image is its next layer's handoff kernel.
- That kernel is enqueued on the same stream, after this layer's wait (`ffn.rs:1266-1270`, and the join half at `:1095`).
- In a pair, the other row writes only its own row of the page.

Hostone writes this as a `// SAFETY:` condition and enforces it: the `&[f32]` over the image must end before the `Cnt` store. That needs a page accessor returning `&[f32]`, in place of the copying `payload_f32_into`.

### 1.4 BatchPort protocol (per layer-batch; eager)

| # | Transition | Who | Waits on |
|---|---|---|---|
| 1 | `Free → Routed(key)`: three DtoH copies (x, ids, w) into the set's pinned buffers, then the set's event (`batch.rs:1054-1090`). | card stream | the route kernels (stream order), PCIe |
| 2 | `Routed(key) → Served(key)`: `set.routed.synchronize()`, then one union call over the pinned x (`batch.rs:1113-1127`). | host | the event (the copies are done), then DRAM and the pool |
| 3 | `Served(key) → Free`: one HtoD copy of the sums (`batch.rs:1135-1162`). The card's post work follows in stream order. | card stream | nothing on the host: it is enqueued after step 2 returns |
| G | The next layer-batch's route is enqueued before this serve (`prefill.rs:1482-1511`). Each set owns its event, so a serve never waits for the next route (`batch.rs:944-963`). | calling thread | nothing |
| F | Failure: poison the tier. There is no release, because nothing waits inside a captured graph (`hybrid.rs:1796-1821`). On a refusal, the refused columns get NaN sums and the error is `name_refusal(read_fault(word))` (`hybrid.rs:1919-1942`). | host | the synchronous fault read |

### 1.5 One owner for each fact

**Fault read.** `fault.rs` gets `read(word)`, which is today's `read_fault`. `Gpu::fault` becomes `self.stream.synchronize()?` followed by `fault::read(&self.fault)`.
- The event-ordered reader survives because it is the general form. A batch service is ordered by its set's event. A stream-ordered read there would also wait for the next route, which is already enqueued.
- The stream-ordered read is just the general read plus one sync. On a fault it costs one sync instead of two (`lib.rs:2253-2278`).
- `fault_or` (`prefill.rs:1920-1924`) then needs no sync of its own.

**Mapped host memory.** There is one `MappedHost`, in `graph.rs`, allocated with `PORTABLE|DEVICEMAP` (the flags `HostFlags` already uses, `graph.rs:358`). It is `Send + Sync`, with words accessed through atomics. `HostFlags` and the step page become views over it.
- `PORTABLE` is needed by the (b) plan's second card, by DSpark's rows on the 3090 (`chain/glue.rs:52, 418, 437, 474` already use `HostFlags`), and by the (b′) expert server.
- The memop builders move to `graph.rs` as well. `plan-triage.md:217` already names `graph.rs` as their owner.

**`capturing`.** `graph.rs:300-307` survives: it is public, and `graph.rs` owns capture. The private copy at `hybrid.rs:679-685` goes.

### 1.6 Where the batch machine lives, and what aa keeps

**What moves.** `Stage`, `Set`, `Exchange` (renamed `BatchPort`), the key-order refusals and `ServeTimes` move to `crates/gpu/src/host/batch.rs`.
- The port is parameterized by `n_used`. Today it hard-codes V4.1's `N_USED = 6` for sizing (`batch.rs:1020, 1026`) and for indexing (`:1061`, `:1117-1127`). `union_scratch` (`ffn.rs:1578-1579`) is parameterized the same way.
- `ExchangeKey {layer, set, at, u}` (`batch.rs:1409-1414`) is not V4.1-specific. It moves with the port as `BatchKey`.

**What aa's crate keeps** (the routing decisions):
- `FfnBatch`'s device buffers and `check_key` (cap and G, `batch.rs:1386-1400`);
- the route and places kernels, `plan_slots`, the key order and the G-ahead enqueue (`prefill.rs:1430-1525`);
- CED;
- the card shadow (`ds41_card_buckets` and the tiles);
- post and join;
- the fault reads at the end of a group (`prefill.rs:1305-1311`);
- the prompt call's records.

`FfnBatch::enqueue_download`, `serve` and `enqueue_upload` (`batch.rs:1357-1382`) become three calls on the tier's batch port.

### 1.7 Rows: S × m

A go serves one unit: m columns at one layer, served by one union call of m columns.

| Chain | Shape |
|---|---|
| `Chain::Step` | S1 × m1 |
| `Chain::Pair` | S2 × m1: two skewed rows, each served on its own because they are at different layers |
| aa's `Rows(m)` | S1 × m, with m ≤ 8 |

**Page layout.** `MAX_ROWS = 2` (`hybrid.rs:430`), `PAYLOAD_OFF` (`:439`) and the word offsets (`:442-445`) become a load-time `PageLayout(S, m, hidden, n_used)`, with the same asserts. Each unit gets:
- `Cnt` and `Lyr` words;
- an image holding m·n_used ids, m·n_used weights and m·hidden x values;
- an hsum of m·hidden values.

**Shared interface.** The per-unit image layout (`HandoffLayout`, `hybrid.rs:776-787`) is where aa's m-column handoff kernel and 03's StepPort meet: the kernel writes it, the port reads it.

This also answers `session-design.md:539`, item 8: the m-row `serve_replay` is `tier.serve_captured(Rows(m))`.

### 1.8 Decisions the types carry

1. **No unwatched tier.** Today a batch refusal on an unwatched tier returns `Ok` with NaN sums (`hybrid.rs:1937-1940`). The engine always watches (`body.rs:2259`). The tiers built without a watch are `deepseek2/mod.rs:440` (no batch path), `gate_deepseek41_chain_ffn.rs:1651, 1654` and `launcher.rs:485`. Once the fault word is a constructor argument, step D deletes the `Ok` path and the clause of `gate_hybrid` that pins it (`gate_hybrid.rs:882`), with a dated coverage reason.
2. **`exclude` goes.** The engine's only caller passes `&[]` (`batch.rs:1120`); only `gate_hybrid.rs:1407, 1459` pass a set. `check_exclude` (`hybrid.rs:2249`), `batch_excluded_slots` and the `excluded_lb` record field go with it (`record.rs:741`, `prefill.rs:1298, 1315`). This is a schema change, so it belongs to step D.
3. **Scratch belongs to the port.** Its width is the port's: narrow for the step port, `UNION_MAX_COLS` for the batch port. The weights belong to the tier.
4. **Each set's stage is an atomic word** (§4.4). Today only one thread uses it, and two atomic stores per layer-batch cost about 0 [derived].

## 2. Decode as a one-column union call

### 2.1 Term by term

T = 32 pool threads. n = the host experts in the call. R = the rows of a pass, counted in units (rows, or r8 groups of 8 rows after r8host).

| # | Term | `serve` today (group engine) | One-column `UnionCall` | Difference | Cost |
|---|---|---|---|---|---|
| 1 | Pool dispatches | 2: the gate/up group (`moe.rs:922`) and the down group with its combines claimed (`:933`) | 2: `gate_up` and `down` with claims (`ops.rs:3810-3827`; `call_dispatches` in `union.rs:146-153`) | none | 0 |
| 1b | The port's `wait_go` | 1 (`hybrid.rs:2187`) | 1 | none | 0 |
| 2 | Lane cut | byte cost (`row_cost`, `ops.rs:2659`), closed form (`:2966-2996`) | tile cost (`tile_cost_of`, `:2686`; `run_row_pool(…, true)` at `:3904` and `:4006`); byte cost for a type with no tile | Each pass has one type in V4.1, GLM, Qwen3.6 and V2-Lite, so the cost cancels and both cuts give `bound_t = ceil(t·R/T)`. | 0 |
| 3 | Home lane | the lane holding the start of chunk t (`chunk_bounds`, `threads/src/lib.rs:498-509`; `ops.rs:3035-3070`) | lane t (`items = nlanes`, `:3022`) | The two are equal whenever ⌊R/T⌋ > R mod T. For all four models R is a multiple of 32: V4.1 gate/up 4608n and down 5120n; GLM 4096n for both; Qwen3.6 1024n and 2048n; V2-Lite 2816n and 2048n. | 0 |
| 4 | Steal block | max(1, lane/4), no cap (`ops.rs:2850`) | min(that, 144/grain) (`:3586`; `UNION_BLOCK_ROWS` at `:2760`) | The cap binds for V4.1 gate/up at n ≥ 5 (lane 144n, block 36n) and down at n ≥ 4 (160n, 40n). For GLM at n = 8 the block drops from 256 to 144. | not priced, sign unknown (more claims against a shorter tail); removed in §2.2 |
| 5 | Claim protocol | load first, then CAS (`DeferredSlots::claim_pass`, `ops.rs:2370-2396`, cheap look at `:2373`) | `fetch_add` on every arrival (`XClaims` `:3441`, `CombineClaims` `:3484`) | about 32 RMWs on one cache line, against about `units` | not priced; removed in §2.2 |
| 6 | Claim units | x: 1 slot (2 if the up reads other bytes); down: n | x: 1 unit (2); down: n experts | none at m = 1 | 0 |
| 7 | Row order | list order | ascending expert id | the lanes cut the same counts | 0 [derived] |
| 8 | Weighted sum | caller, list order (`moe.rs:940-944`) | caller, list order (`sum_col`, `ops.rs:3219-3235`; `UNION_INLINE_COLS` `:3155`) | two binary searches per listed expert | 0 … +0.1 µs per call [derived] |
| 9 | Per-call setup | 3n views resolved (`moe.rs:790-797`) | the union plan (`moe.rs:983-1059`) | similar work | ≈ 0 [derived] |
| 10 | x | copied from the page into `Hybrid.x` (`hybrid.rs:2099`, `1412`) | read in place as a `Tensor2View` (invariant in §1.3) | one 20,736 B copy disappears (V4.1) | −0.3 … −1.5 µs per layer-step [derived] |
| 11 | Stats store | `LAST_QUANT_SLOTS` (`ops.rs:1644`) | none | one relaxed store | ≈ 0 |
| 12 | Deferral lever off | 2 (the pre-pass runs inline under `QUANT_INLINE_COLS`, `:2271`) | 4 (`union.rs:146-153`) | the oracle arm only | not on the default path |

### 2.2 The two differences hostone removes

(a) **Steal-block cap.** `UnionRows::block_cap` returns `None` for a narrow call. The block rule then becomes one countable formula:
- narrow: `max(1, lane/4)`;
- wide: `min(max(1, lane/4), 144/grain)`.

The prompt's narrow calls (the tail of a group, 8 or fewer columns) follow the same rule. Their bits do not change, because a row's value does not depend on which thread computes it (`ops.rs:2764-2768`).

(b) **Cheap look.** The `XClaims` and `CombineClaims` claim passes load the counter first, and break once it is past the unit count. `DeferredSlots` already does this (`:2373`), and so do the lanes (`:3074`).

After (a) and (b), the one-column call differs from `serve` only in rows 7–11. Each of those is about 0 or negative.

### 2.3 Row steps with m ≤ 8

**Today.** A pass of m rows serves each (layer, row) as its own one-column call (`serve_one`, `hybrid.rs:1997`). That is 2m dispatches plus m `wait_go` jobs.

A single group-engine call over m rows would not help. It passes `MAX_DEFER_SLOTS = 16` (`ops.rs:2278`) once 6m > 16, which for V4.1 is m ≥ 3. Past that it falls back to a combine on the caller, which costs +0.19 ms per row (rig-log 2026-09-24, ktok section, the 5x4 arm; derived there from the measured arms).

**Union.** One call of m columns issues 2 dispatches for m ≤ 8 (`UNION_INLINE_COLS = DEFER_MAX_COLS = 8`, asserted at `ops.rs:3160`).
- The distinct experts number at most m·n_used, which is ≤ 64 = `UNION_CLAIM_EXPERTS` (`:3150`).
- Each distinct expert's rows are read once for every column that lists it.

**Bytes.** When the lists are disjoint, the host reads m times one row's bytes. That case was measured as linear: 24.56–24.89 ms per row (rig-log 2026-09-24, arms `engine:5` to `5x4`). What the union saves is the overlap between rows. For the pair, `overlap_slots / pair_row1_slots` counts it (`hybrid.rs:1182-1190`); this memo cites no value for it.

**Proof.** The group engine has no counterpart for m > 1, so the identity claims below cover m = 1 only. m > 1 is a new shape, proven by two things:
- the existing bit clause: k columns equal k one-column calls, for k = 1, 2, 3, 6, 8, 9 and both arms (`tests/union.rs`);
- a dispatch count of 2.

### 2.4 Structural tests hostone adds

All are synthetic. `just gate-ops` runs them through `--lib`, except T6.

- **T1, dispatch count.** A narrow call issues 2 pool dispatches. This is counted both by the scratch's pass counter and by the pool's `dispatches` (the `dispatched` helper, `union.rs:158-166`). Expected otherwise: 0 when every list is empty, 4 on the oracle arm, 5 past 8 columns. The count is taken at the `UnionCall::run` / `HostLayer::experts` level; the port's `wait_go` adds 1 outside it.
- **T2, lane bounds.** `run_row_pool`'s cut becomes a pure function `lane_bounds(costs, T)` (from `ops.rs:2966-2996`). It must equal `ceil(t·R/T)` and also a walk over the rows. It is checked on the V4.1, GLM, Qwen3.6 and V2-Lite shapes, for n = 1..8, T ∈ {1, 2, 16, 31, 32, 64}, grain 1 and 8, and both cost models.
- **T3, home lanes.** Under the row cut, participant t starts on lane t whenever ⌊R/T⌋ > R mod T. The test prints any shape where the condition fails.
- **T4, block rule.** Each lane's block equals the formula in §2.2(a), on the same shapes.
- **T5, claims.** A `cfg(test)` counter sits beside the claim state. A pass makes at most `units` plus (the number of arrivals before the counter passed `units`) RMWs, and an arrival after that makes none. This is the bound `DeferredSlots` already meets.
- **T6, allocations.** A narrow call makes zero allocator calls after load. This test joins `tests/alloc.rs` and runs under `just gate-alloc`.
- **T7, bits.** m columns equal m one-column calls (this already exists in `union.rs`). Step C adds an md5 pin of one-column outputs over a fixed synthetic set, recorded against `experts_into` while it still exists. The pin stays after step D as the record of what the deleted path computed.

### 2.5 Predicted Δ, and the proof step C lands on

**Prediction.** Rows 7–11 of §2.1 sum to −0.2 … −1.5 µs per layer-step. Over 40 layers that is −0.01 … −0.06 ms per V4.1 step of 31.6 ms, or −0.03 … −0.19 % [derived].

**The card is refused.** An `ab` card with that band is refused as `CARD_UNDER_RULER` (`tools/ref/card.py:68`; the kinds are `ab|profile|calibration|baseline|exclusive`, `:95`). So the 03 plan's "one decode A/B" for hostone cannot run with today's card kinds.

**Proof.** Step C therefore lands on path (i): T1–T7 plus the bit-identical gates. Whether to add a non-inferiority card kind is the lead's call (`cpu.md` CPU6, option (ii)).

### 2.6 What goes and what stays

These go in step D, after step C:

| Item | Where | Grep, and today's callers |
|---|---|---|
| `moe::serve` | `moe.rs:910-953` | `grep -n 'serve(' crates/model/src/moe.rs`: its only caller is the free `experts_into` |
| `HostScratch` | `moe.rs:853-899` | `grep -rn HostScratch crates`: `ffn.rs:1510, 1536`; `deepseek2/mod.rs:93, 105`; `tests/ds41_host.rs:63, 242, 266, 347`; `gate_deepseek41_chain_ffn.rs:135, 878, 905, 1172`; `gate_deepseek41_step.rs:2446-2447` |
| Free `experts_into` and `check_host_call` | `moe.rs:767-805`, `817-846` | `grep -rn 'experts_into' crates --include=*.rs`: `deepseek2/mod.rs:128` switches to the free `experts_union_into` (`moe.rs:1406`) with one column |
| `HostLayer::experts_into` | `moe.rs:1536-1570` | `ffn.rs:1590`, `tests/ds41_host.rs:274, 388`, the step gate; replaced by `HostLayer::experts` |
| The two-method `HostExperts` | `hybrid.rs:1116-1146` | stubs at `launcher.rs:457` and `gate_hybrid.rs:770-807` |
| `Hybrid.x` | `hybrid.rs:1412, 1465, 2099` | none |
| `matmul_q_group_cols_into`, `Entry` (only `Plain` would be left), `wide`, `Arm::tile_lanes`, and the wide pre-pass's combine arm | `ops.rs:1185-1219`, `1380`, `2548` | `grep -rn 'matmul_q_group_cols_into\|Entry::Cols' crates`: `bench_v41_host.rs:1014, 1028, 1336` |
| `GroupInput::Cols` / `Quantized`, `QuantizedCols` | ops.rs | only if the grep shows no caller after benchprune (check `tests/ops.rs`) |
| The `exclude` path and the unwatched batch path | §1.8 | `batch.rs:1120`, `gate_hybrid.rs:882, 1407, 1459` |

Two notes on the table:
- `run_row_pool`'s `tile_lanes` parameter stays; the union needs it.
- The bench's engine arm (`bench_v41_host.rs:1587-1626`) is a second copy of `serve`. It moves to the union in step C.

**The `ds41_host` test.** It reads the f32 combine through `HostScratch::par` (`tests/ds41_host.rs:291, 406`), and the union never keeps an f32 combine. The replacement clause compares the union's quantized combine slot bytes with `quantize(swiglu_clamp(gate, up))`, bit for bit, where `gate` and `up` come from accessors on the gu slab. FAIL-first check: drop the clamp from the combine and the clause goes red.

**What stays.** The V2-Lite CPU engine still calls `matmul_q_group` / `_swiglu` (`ffn.rs:71`; `moe.rs:535, 688`; `arch/deepseek2/attn.rs:128`). So these stay:
- `group_core`, `run_group`, `DeferredSlots`, `quantize_slots`;
- `matmul_q_group`, `_into`, `_swiglu`, `_swiglu_into`, `_batch`, and `matmul_q`;
- `BLOOMERY_DEFER_QUANT`, in the form hostcfg leaves it (a test-only oracle arm);
- `set_defer_quant` (`tests/union.rs:417, 677, 716, 932`).

They go with CPU1(a). This corrects the "얻는 것" list of `cpu.md` CPU6: `DeferredSlots` and the lever do not leave with hostone.

## 3. The next models

### 3.1 GLM-5.3-Flash UD-Q4_K_XL

**Shape.**
- 288 experts, top 8, expert ff 2048, hidden 4096, routed SwiGLU clamp 10.0 (`modelspec-design.md:499-525`).
- Layer 45 is the nextn layer (`:535-536`), so the decode trunk is layers 3–44: 42 MoE layers.

**Routed types** (`:9-15`):
- gate/up: Q4_K, except Q5_K on layer 11;
- down: Q5_K on layers 3–10, 13–43 and 45; Q6_K on layers 11, 12 and 44.

**Host kernels for these types.**

| Type | qdot kernel (`qdot/src/lib.rs:134-146`) | Activation (`:204`) | Tile (`:443-453`; `ops.rs:2668-2675`) | r8 row-lane |
|---|---|---|---|---|
| Q4_K | yes | q8_2_x4 | yes | no (r8 is Q3_K only, r8host `r8file.rs:277`) |
| Q5_K | yes | q8_2_x4 | yes | no |
| Q6_K | yes | q8_2_x4 | **no**: `dot_row` once per column | no |

What is missing:
- A Q6_K tile. It matters only for the prompt's down pass on layers 11, 12 and 44, where each row is unpacked once per column instead of once per 8 columns. Decode has one column and is unaffected.
- A Q4_K r8 layout, needed only if r8 becomes V4.1's default.

**Placement.** Three facts from the code:
- `CardFormat::of_routed` refuses Q5_K (`placement.rs:367-369`).
- A layer is eligible for card experts only if every routed stack in it has a card format (`:1188-1199`).
- So 40 of the 42 trunk layers carry a Q5_K stack and can hold no card expert.

Layers 12 and 44 are eligible by format, but it is undecided whether a kernel reaches them:
- V4.1's card experts are Q3_K gate/up tiles plus `q4k_gemv_tiles`.
- The qwen3moe program does run Q4_K and Q6_K routed stacks (`modelspec-design.md:699`, `:778`).
- Whether GLM's program reaches a Q6_K down is the act-planes `card_reads(ty, Family)` question.

This memo counts those two layers' bytes as neither card nor host. Decode is host-served on at least 40 of 42 layers and, at today's placement, on all 42.

**Bytes.**
- Per expert: 15,204,352 B (Q4_K 4,718,592 × 2 + Q5_K 5,767,168).
- Per expert on layers 12 and 44: 16,318,464 B. On layer 11: 18,415,616 B.
- These are [derived] from the dims at 144, 176 and 210 B per 256 values.
- Per token, with all 42 layers host-served: 8 × (39 × 15,204,352 + 2 × 16,318,464 + 18,415,616) = **5,152,178,176 B** [derived].

For comparison, V4.1's all-host bytes are 4,043,243,520 B per token: the measured byte count of the `engine:6` arm (rig-log 2026-09-24). V4.1's own decomposition, 6 × (38 × 16,773,120 + 2 × 18,247,680), gives exactly that number. So GLM is 1.274× V4.1.

At 135.4 GB/s (`engine:5`, same source), that is a **38.05 ms per token host floor** [derived].
- This assumes the Q4_K/Q5_K × q8_2_x4 dots stay DRAM-bound, as V4.1's Q3_K × q8_K dots do.
- They do fewer MACs per byte (1.78 and 1.45, against Q3_K's 2.33), but their host rate has not been measured.

Two more numbers:
- The nextn layer, if host-served, adds 121,634,816 B per draft pass [derived].
- The routed bytes in the file are 288 × 659,226,624 = 189.9 GB, against V4.1's 258.8 GB [derived]. Capacity is not a new constraint.

**The decision.** Beyond this host tier, GLM needs exactly one more thing: a card routed-Q5_K expert kernel, that is, a decode gemv. The prompt already has `gemm_q5k`, which the user decided to keep on 09-27. Without it the card holds almost no GLM routed expert.

**Do the page and slot sizes hold?**
- `MAX_USED = 16 ≥ 8` (`hybrid.rs:449`, `:451`): yes.
- `EXPERTS_INTO_MAX = 8` (`moe.rs:752`): yes, exactly at the bound.
- `UNION_CLAIM_EXPERTS = 64 ≥ 8·8`: yes, exactly at the bound for m = 8.
- The image is (64 + 4096) words = 16,640 B per row, against V4.1's 20,736 B [derived; `hybrid.rs:1004-1006`].
- `MAX_ROWS = 2` does **not** hold a host-served verify with m > 2 (§1.7).
- The transport's `N_USED = 6` (§1.6) must become the shape's `n_used`.

### 3.2 Qwen3.6-35B-A3B

**Shape.** 40 MoE layers, 256 experts, top 8, expert ff 512, hidden 2048 (`modelspec-design.md:453-468`). Routed Q5_K appears as the down on layers 0, 2–33 and 35–37, and as gate/up on layer 1 (`:763`). The other routed stacks are Q4_K and Q6_K (`:778`).

**Bytes.**
- Per expert (Q4_K gate/up plus Q5_K down): 1,900,544 B.
- All-host per token: 0.606–0.615 GB [derived]. The four downs that are not Q5_K are bounded between their Q4_K and Q6_K sizes.
- That is about 4.5 ms per token at 135.4 GB/s.

**Capacity.** Neither card needs a host tier for capacity.
- The A6000 (48 GB) holds the 22.4 GB file whole.
- The 3090 (24 GB) holds it with about 2–3 GB to spare [derived]. KV takes 20,480 B per position over the 10 GQA layers at f16, which is 0.67 GB at 32K. The GDN and conv state take 65.9 MB per slot (`:769-773`).

**What puts experts on the host today** is `of_routed`: the 37 layers that carry a Q5_K stack, about 0.56 GB per token [derived]. The qwen program is whole-card (`engine.rs:36`) and has no host tier, so nothing runs those stacks (`:763`).

**What it needs** is the same card routed-Q5_K kernel as GLM, not a host tier. With that kernel it runs whole-card on the A6000, which is what `session-design.md:73` assumes.

## 4. Resource timelines

### 4.1 V4.1 decode, per step

The row is from `nsys-ds41-d6-n8-101249.sqlite`: A6000, plan (a), router-frequency list, depth 6, n 8 (`plan-triage.md:195`; `plan-ledger.md:1159`).

| Resource | Per step | Source |
|---|---|---|
| Card critical path | 8.56 ms | measured |
| Bridges (go to join, 40 layers) | 22.61 ms | measured |
| Idle between replays | 0.42 ms | measured |
| Step | 31.6 ms; the wall is the sum, because the card waits inside each bridge | measured |
| PCIe per layer | image DtoH plus the hsum read, about 41 KB, about 1–2 µs | [derived] |
| NVMe | 0 at steady state: the host set is populated and locked | [derived] |

Another fit, from a different nsys row (kr-moeattn, `plan.md:78`), gives the bridge as 36.0 + 119.5·k_host µs per layer, for k_host 2..6. It is not combined with the row above.

**What the redesign changes.**
- StepPort, card side: nothing. The memops (`hybrid.rs:1064-1093`), the handoff nodes and the page are unchanged, and so are the node counts.
- StepPort, host side: row 10 of §2.1 goes, −0.3 … −1.5 µs per layer-step. Rows 4 and 5 are 0 after §2.2.
- Tier: nothing on the success path. A fault read saves one sync.

The bridge is the critical path. The redesign shortens it by 0.01–0.06 ms per step and adds nothing to it [derived].

### 4.2 V4.1 prompt, per layer-batch

The setup is the lcg prompt, P 4096, G 2, on the A6000 (rig-log 09-26#prefillgroup-ab, via the table in GC3).

| Resource | What | ms |
|---|---|---|
| Calling thread | enqueue, then its share of the union | 4.3–4.4 (measured) |
| Host workers | the union | 68.6–69.3 (measured) |
| Card | card_out + card_in, under the union | 21.9 + 11.7 = 33.6 [derived sum] |
| Host wait | the route event | 0.83 (measured) |
| Wall | the host's sum; the card is in the shadow | ≈ 74 [derived] |

**What the redesign changes.** BatchPort adds nothing on either side.
- Card: the same three copies and one event per download (`batch.rs:1080-1087`), and one copy per upload.
- Host: the same union call. The move adds no work to any transition.

No term of the redesign is on the critical path.

### 4.3 GLM decode [derived]

The per-layer bridge is about 36.0 + 8 × 119.5 × 0.9065 ≈ 903 µs. This is V4.1's fit from the kr-moeattn nsys row (`plan.md:78`), scaled by bytes per expert. Over 42 layers it gives 37.9 ms per token, which agrees with the byte floor in §3.1.

The KDA mixer's card time comes before each route and adds to that, because no shadow hides it (`session-design.md:77`).

### 4.4 The flow levers

These are hypotheses. This memo says only whether the design helps or blocks each one.

**Route as a graph** (up to −4.3 ms per layer-batch, per GC3): neutral.
- The download's copies and its event record capture as graph nodes.
- The stage transitions are host-side at issue time either way.
- The blocker GC3 names (CED widths create many graph keys) is untouched.

**Issue on an SMT sibling** (GC3; `launcher.rs:255-310`): events do not block it.
- A sibling thread can issue the next route during the union, and can issue a batch's upload and post once its serve ends.
- What blocks it is ownership. `Exchange` is one `&mut` object inside `FfnBatch` (`batch.rs:951`), which ties issue and serve to one thread.
- BatchPort helps, provided each set's stage is an atomic word (§1.8 item 4).
- A later flow round, with its own card, could go further with a flag-gated upload: a stream waits on a mapped "served" word, in the style of exllamav3's `cuStreamWaitValue32` (`moe_handoff.cu:122-140`). The issuer could then enqueue the upload and post at route time. This needs StepPort's memops on the eager path and one stream per set, because a wait holds its stream.

**(b′):** a third port over the same tier. It needs the `PORTABLE` page (§1.5).

## 5. References

| | ik_llama.cpp `c10fbbcc` | mistral.rs `d5ae0f1` | exllamav3 `0740edc` | This memo |
|---|---|---|---|---|
| Placement grain | a layer's expert stacks: `-ncmoe` overrides `blk.i.ffn_(up\|down\|gate\|gate_up)_exps` to the CPU buffer type (`src/llama-load-tensors.cpp:275-288`) | whole layers: `LayerDeviceMapper::map` moves the activation to the layer's device (`device_map/mappers.rs:49-52`, `peer.rs:26-31`) | per expert inside a layer: "the assignments divide between GPU-resident and CPU-resident experts" (`modules/block_sparse_mlp.py:845-853`) | per expert (`SlotMap`, `hybrid.rs:288-400`) |
| Host compute | one op per projection whatever the token count (`ggml.c:18183` `mul_mat_id`, `:18502` `_up_gate`); the weighted sum is a separate op (`ggml_mul_multi_add` `:6349`, `iqk_mul_multi_add` `:26499`) | a repacked per-expert `indexed_gemv` first (`gguf/cpu.rs:28-66`); otherwise it dequantizes every expert and gathers (`:69-83`) | a child process owns the expert weights and reads a job ring in pinned shared memory (`model/moe_cpu_host.py:21-28`) | the union over a pinned thread pool (`ops.rs:3712-4048`) |
| Hand-over | the scheduler copies a split's inputs behind a per-backend event or a backend sync (`ggml-backend.cpp:2016-2050`). For host-resident expert weights on the prompt, it reads the ids back (`:2083`) and uploads only the runs of active experts (`copy_experts`, `:2107-2138`) | `Tensor::to_device` at a layer boundary | on the stream: D2H copies, a publish (`cuStreamWriteValue32` or a flag kernel), a GEQ wait with an abort word, then the H2D readback (`moe_cpu_host.py:24-28`; `cpu/moe_handoff.cu:81-91, 108-140`) | Step: memops in a captured graph. Batch: events plus a host sync. |
| In-graph host service | none: an event wait on a non-CUDA backend is `#if 0`, then `GGML_ABORT` (`ggml-cuda.cu:5268-5284`) | none: no memop or host-func calls; graphs are decode-only (prefill is eager, `pipeline/normal.rs:1787, 1821`) and are disabled for good after a capture or replay error (`:2247-2256`) | none: the handoff is eager, and nothing on this path captures a graph | StepPort |

**What this memo follows.**
- StepPort uses exllamav3's mechanism, but captured in a graph. None of the three does that.
- BatchPort uses ik's scheduler shape: events and a host sync, eager.
- The union follows ik's "one op per call", with the sum done on the caller.

**Where the three differ from us.**
- Only exllamav3 splits a single layer's experts between card and host.
- On the prompt, ik computes host-resident experts on the card by uploading the active ones. We compute them on the host, with the card's own experts running in the shadow.

## 6. Migration

### 6.1 Steps

| Step | What | Class and proof | Touches the CPU dispatch path? | Gates (bit for bit) |
|---|---|---|---|---|
| prerequisites | r8land; hostcfg (levers to `HostCfg`, `DEFER_QUANT` becomes a test-only oracle, `STEAL_BLOCKS` a const 4, `Chain` moves to `model.rs`, `REPLAY` becomes an argument); benchprune | as in the 03 plan | hostcfg only (the lever becomes a const), by its own proof | their own |
| A | one owner per fact (§1.5) | move: `just ptx-scan` of `generate_ds41` and `gate_e2e` identical; node counts identical; eager = replay | no | gpu-hybrid, ds41-step, ds41-prefill, ds41-faults, ds41-chain-ffn, gpu-e2e |
| B | `HostTier` / `StepPort` / `BatchPort`; the batch machine moves; the residency moves (option A); the stats view | move, plus: `--records-schema` byte-identical, `records-refresh` leaves `git diff` empty, `--plan` identical | no | as A, plus `stage-gpu-load-v41` if the residency moves |
| C | decode as a union call (§2) | dispatch path, proven by T1–T7; **no A/B** (§2.5) | **yes**: decode changes engines. Proof is T1–T7 plus the bits; the card is refused | union, ds41-host, ops, moe, alloc, then as A; bench `--check` |
| D | deletions (§2.6, §1.8 items 1 and 2) | delete: the grep finds no caller; ptx-scan equal; dated coverage reasons; the `excluded_lb` schema change with its readers | **yes**: removing `Entry::Cols`, `wide` and `Arm::tile_lanes` edits `group_core`, which the V2-Lite CPU engine runs. The removed branch is unreachable (grep), and it costs one constant compare per group call, under any ruler [derived]. Proof: gate-ops, gate-moe, gate-ffn and gate-forward bit-identical. | as C, plus the changed gate-hybrid clauses |
| E | hybridgate | clause move, FAIL-first | no | gpu-hybrid |
| later | StepPort S × m, together with aa's `step_rows`; the atomic stage and flag gate (a flow round) | node pins, or its own card | no | theirs |

### 6.2 Order

1. r8land.
2. hostcfg (after aa's levers2) and benchprune.
3. Step A.
4. Step B, after aa's `gpumodel` (which owns `model.rs`).
5. Steps C, D, E.

All of this lands before aa's DS6, layerprog and batchwide, which then build on `BatchPort`. Later, `modelspec` supplies `HostLayerSpec` from `ModelSpec`, and `step_rows` shares `HandoffLayout`.

### 6.3 Files each step touches

| Step | 03's files | aa's files |
|---|---|---|
| A | ~~`graph.rs`, `fault.rs`, `lib.rs`, `hybrid.rs`~~ `fault.rs`, `hybrid.rs` | ~~none, as long as `HostFlags` keeps its API (`glue.rs` is not touched)~~ `graph.rs`, `lib.rs` (corrected on aa's signature, 2026-09-27: both are aa's files); `glue.rs` is not touched as long as `HostFlags` keeps its API |
| B | `hybrid.rs` → `host/*` | `batch.rs` (`mod exchange` moves out), `ffn.rs`, `body.rs`, `prefill.rs`, `model.rs` (option A); also `deepseek2/mod.rs`, `record.rs`, `generate_ds41.rs`, and the gates |
| C | `ops.rs`, `moe.rs`, `host/step.rs`, the tests, `bench_v41_host.rs` | `ffn.rs` (`Ds41Host`), `deepseek2/mod.rs` |
| D | `moe.rs`, `ops.rs`, `host/*`, `gate_hybrid.rs` | `record.rs`, `generate_ds41.rs`, `tools/bloomery/records.py` |

### 6.4 Where r8land can change this memo

1. Units and the cap become 18 r8 groups. The V4.1 thresholds are still n ≥ 5 [derived].
2. `HostLevers.r8` moves into `HostCfg`.
3. `HostResidency` becomes `HostR8` plus `HostSet::of_r8`. The tier's field type changes, and so does the drop order against `Arc<Sidecar>`.
4. `Ds41Host::build` changes.
5. New record fields appear, so step B's schema baseline is the tree after r8land.
6. r8host adds 118 lines to `union.rs`. The m = 1 bit clause must still cover both layouts afterwards.
7. The r8 decode coverage moves from the group path to the union path.

## 7. What aa signs

1. The batch machine moves to `crates/gpu/src/host/batch.rs` as `BatchPort`, parameterized by `n_used`. `FfnBatch` keeps its device buffers, `check_key` and the routing decisions.
2. `ExchangeKey` moves with the port as `BatchKey`, with the same fields.
3. `fault::read` becomes the only fault reader. `Gpu::fault` becomes one sync plus that read, and `fault_or` drops its own sync.
4. `MappedHost` and the memop builders move to `graph.rs`, with `PORTABLE|DEVICEMAP` and `Sync`. `HostFlags` becomes a view over it and keeps its API.
5. `graph::capturing` is the only `capturing` left.
6. `HostResidency` moves from `GpuModel` into `HostTier` (option A). That changes the load order in `model.rs:528` and the `ChainBody` load signature.
   - Option B: it stays in `GpuModel`, and the tier holds nothing of it.
   - Nothing on the service path reads the residency; it is a load artifact with a drop-order rule. The single-owner argument for option A is organizational, not functional, so either is acceptable.
7. The fault word becomes a constructor argument of `HostTier`, and the unwatched `Ok` + NaN path goes.
8. `exclude`, `batch_excluded_slots` and `excluded_lb` go in step D (a schema change).
9. The per-unit `HandoffLayout` (S × m) is the shared interface with aa's `step_rows` kernel.
10. `HostServed` means `tier.serve_captured(chain)`. `HostExperts` means compute only.
11. Order: steps A and B after gpumodel; steps C, D and E before DS6, layerprog and batchwide.
12. Step C lands without a timed A/B, on T1–T7 plus the bit-identical gates, because its card falls under the ruler and is refused.
13. Finding for aa's GLM plans: at today's placement GLM decode is all-host (about 38 ms per token [derived]). Its one missing piece is a card routed-Q5_K expert kernel, which Qwen3.6 needs as well.

**aa's signature (2026-09-27).** aa signed items 1–11 and 13, and item 12 with a condition. The conditions, with
03's answers where aa asked for a decision:

- On steps A and B: `graph.rs` and `lib.rs` are aa's files (§6.3 corrected), and `del2` has just changed both. Step A
  rebases onto `del2`. Before A lands, and again before B lands, 03 sends aa the list of hunks in aa's files for review,
  the same shape as the act-planes condition on its item 9. B's aa files are `batch.rs`, `ffn.rs`, `body.rs`,
  `prefill.rs`, `model.rs`, `deepseek2/mod.rs`, `record.rs`, `generate_ds41.rs` and the gates.
- One step at a time on aa's files. Act-planes' step B and this memo's step B both wait for `gpumodel` and touch the
  same aa files. **03's order: act-planes B first**, since it is ready right after `gpumodel`, while this memo's B also
  waits for hostcfg, benchprune and step A. This memo's B follows it.
- hostcfg against `gpumodel`, which rewrites `model.rs` and launches right after `del2` lands. aa asked whether
  hostcfg's two `model.rs` hunks (the `enum Chain` move, `REPLAY` as an argument) land before `gpumodel`'s base is
  cut, or go to `gpumodel` as named hunks. **03's answer: neither hunk stays in hostcfg.**
  - `Chain` stays in `hybrid.rs`, and `gpumodel` imports it where `HostServed` names it. Step B moves it, with the step
    port, to `host/step.rs`: the chains are the step port's.
  - `REPLAY` does not become an argument. It goes. After `del2` the replay watch carries nothing: its one
    construction (`GpuModel::replay_graph` on `del2`) passes `failed: None`, and `first_serve_lag_ns` has no reader.
    `gpumodel` drops the `serving_replay` wrap when it rewrites `replay_graph`. It also takes the `hybrid.rs` side of
    the deletion as named hunks: `ReplayWatch`, `REPLAY`, `serving_replay`, the `watch` parameters of the two service
    functions with the lag they add, and `HybridStats::first_serve_lag_ns`. It sends that hunk list to 03 before
    landing. The lint count then does not rise between the two sides.
  - So hostcfg edits no `model.rs` line and can fly beside `gpumodel`. Its hunks in aa's files go to aa as a list before
    it launches: the `BLOOMERY_HYBRID_OVERLAP=0` branch in `chain/ffn.rs`, and any construction site the typed config
    changes. A hunk in a file `gpumodel` rewrites waits for `gpumodel`.
- On 6 (option A): `gpumodel` keeps `host: Option<HostResidency>` as it is today and builds no new API around it.
  Step B moves it, and `host_residency()` becomes a `HostServed` method in B, so `gate_load_v41`'s path keeps working.
  B's landing batch runs `stage-gpu-load-v41` and `stage-gpu-load-v41-lock` by name.
- On 10: `gpumodel` defines `HostServed` as one method, `serve_captured(&mut self, chain: Chain)`, in place of
  `serve_replay` and `serve_replay_pair`, so the name matches this memo's. The body holds the tier object.
- On 8: `excluded_lb` is a field of the `stat prefill lb` kind, which `tools/flow/ds41_prefill.py --counts` can read.
  Step D changes `tools/bloomery/records.py` and `ds41_prefill.py` with it, and lands with `gate-gpu-ds41-flowcounts`
  green.
- On 9: `HandoffLayout` has one owner, `host/step.rs`. aa's handoff-kernel constants are tied to it by a compile-time
  assertion or a unit test.
- On 12: aa agrees that T1–T7 plus the bit-identical gates are step C's landing proof.
  - This is also where AGENTS' "a round that touches the dispatch path ends with a same-lease A/B" meets `card.py`'s
    ruler refusal. So before C launches, the user is asked whether to add a non-inferiority arm. Its margin would be
    the 4-round ruler; it would ride on a V4.1 sitting already scheduled, with no box time of its own.
  - §2.2 (a), removing the steal-block cap, also reaches the prompt's narrow tail calls, and (b), the cheap look, also
    reaches the wide passes; §2.5 counts decode only. So step C's prediction adds the prompt terms at P = 512 and 4096:
    the RMWs saved per wide pass times the passes, and the tail calls per layer-batch with their block change.
    Otherwise (a) is limited to `Chain` calls. Prefill is a headline metric.
- On 13: the card routed-Q5_K expert gemv enters wave 5 as the first kernel of aa's `opslib`, common to both next
  models. aa adds it to `rebuild.md`'s wave-5 row. The triage's hosttier item names that owner.
