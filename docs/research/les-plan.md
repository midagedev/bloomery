# les-plan — LES ideas on the NVMe/host expert tier's decode, for 0.3.3 (design + experiment plan)

This is a design and experiment plan, with no engine code and no box run. It asks which ideas from LES (Lookahead Expert
Scheduling, [tc-mb/llama.cpp-omni `docs/les/lookahead-expert-scheduling.md`][les]) give a measurable decode speedup on
our paged-expert path, and in what order to build and measure them. Tree read: main `d90151ed`. Tags follow
`tierengine.md`: [measured, source] a printed number; [derived] arithmetic from code and measured terms;
[assumption] stated, to be checked. "Fill" = one expert's parts read into its arena slot; "f" = decode fills a token;
"L" = layers a token with ≥ 1 miss; "QD" = reads in flight to the drive.

**LES as published.** These are their claims on MiniCPM-V 4.7 35B-A3B, iPad Pro M4 16 GB; none is ours. The baseline
reads experts from SSD at 2.97 tok/s. Stage 1, "lookahead expert scheduling and load optimization, including weight
layout, fewer read calls, and direct writes into GPU-visible memory", reaches 9.81 tok/s. Stage 2, layer-wise mixed
precision, reaches 14.92 tok/s. Stage 3, DSpark speculative sampling, reaches 17.39 tok/s. Their mechanism, in their
words: "Cross-layer routing prediction loads the experts the next layer is likely to need ahead of time. Experts that
the actual router selected and that are already loaded are computed first, while the remaining required experts are
loaded in parallel"; "Prediction is used only for prefetch and scheduling". The page gives no recall, no per-term
breakdown and no read sizes.

**Decision summary.**
1. On our box, the CPU computes the paged experts. LES stage 1 therefore splits into three levers, in this order:
   - C1: the one-column step computes its resident experts first.
   - C2: a fill's three parts are read at once.
   - C3: a cross-layer prefetch.

   The other parts of stage 1 predict about 0 here: repacking (C2 already gets its queue depth),
   fewer read calls, and GPU-visible writes (GB10 only, S7).
2. Stage 2 (mixed precision) is not lossless against the shipped file: it is a different model file. Stage 3 (draft)
   already exists for Qwen3.8 (MTP) and Qwen3.6 (DSpark), and its multi-column steps already take C1's path. Both are
   maintainer questions (§5), not 0.3.3 work.
3. C3 opens only on a measured miss recall. A free Mac replay and an oracle-prefetch ceiling run come first (§3).

## 1. Current state

**Does decode compute resident routed experts first while misses load? It depends on the chain.**

- **One-column step (`Chain::Step`, a plain decode with the draft off): it waits for every miss.** The step port
  serves the layer through `experts.experts_into` (`crates/gpu/src/host/step.rs:1050`), then
  `HostRun::experts_into` (`crates/gpu/src/host/run.rs:215-229`), then `HostLayer::experts_into_on`. That function
  claims and fills before any compute: `ensure_pinned(…, Claim::Step)` (`crates/model/src/moe.rs:2600-2610`) runs
  `NvTier::fill_missing` (`crates/gpu/src/host/nvtier.rs:2211-2248`), which calls `fill` on the compute pool
  (`threads::pool().for_each_chunk`, `nvtier.rs:2253-2266`). The fill is joined before it returns, and only then does
  the compute start (`moe.rs:2611-2632`). The card waits on the host signal (`step.rs:1063`), so the fill wall sits on
  the token's critical path.
- **Multi-column step (`Chain::Cols(m)`, m = 2..`DEFER_MAX_COLS`: a draft's verify columns): it computes resident
  experts first.** This landed in `4d6dbeb2` (2026-10-12) and is in main. `serve_one` sends `Cols` to `serve_cols`
  (`step.rs:985-986`), then to `experts_step_union_into` (`step.rs:1177`). That path issues the misses as a reader job
  (`prefetch_pinned`, `moe.rs:2777-2779` → `NvTier::prefetch` → `Inflight::queue`, `nvtier.rs:2275-2300`) at
  `Kind::Step`, rank 0, ahead of Prompt and Victim (`crates/gpu/src/host/nvtier/reader.rs:37-60`).
  `serve_step_split` (`moe.rs:1850-1939`) then splits the layer's ids by "slot already filled" (`:1860-1865`),
  computes the hits group, waits on the ticket (`:1912-1921`, profiled as `host_fill_wait`), computes the late group,
  and sums once in list order (`sum_batched`, `:1937`). The bits do not depend on the compute order. A hit-only layer
  keeps the single path (`:2781-2787`).
- In both chains **a fill reads its three parts serially on one thread**. `read_pick` issues three `read_into` calls,
  one after another (`nvtier.rs:2354-2373`). The reader's unit is a pick, not a part (`reader.rs:1-3`, `:24`). A layer
  with one miss, the common case at small f, therefore reads at QD 1. The reader has 16 threads
  (`NVTIER_FILL_THREADS`, `nvtier.rs:107`).

**Is gate lookahead built? It is plan only.** It is designed in `tierengine.md` §3 (c) (`:130-176`), and its step is
S5, "Not in main" (`:497`). A search for `lookahead` under `crates/` finds only the tokenizer. The engine has no
`Kind::Prefetch` (`reader.rs:37-47` has three kinds). No predictor exists, router-based or table-based. The route
trace (`crates/gpu/src/host/route_trace.rs:1-12`) records each one-row step's ids and slots (`C`/`T`/`H`), not the
residual. So a trace can score an id-based predictor offline, but not a router-on-activation one.

**Expert on-disk layout and read pattern.** Each layer's routed experts are the GGUF's three stacked tensors, gate, up
and down, each `[…, n_expert]`, possibly in different shards (`nvtier.rs:1363-1386`). Expert `id`'s part `p` is one
contiguous range at `data_base + offset + id × per` (`nvtier.rs:1409-1418`). The three parts of one expert therefore
lie in three distant ranges, one per tensor, about `n_expert × per` apart [derived]. Experts `id` and `id+1` of one
part are adjacent. Each part is read with its own `O_DIRECT` pread, aligned to 4 KiB (`aligned_span`). It lands in its
own window of the slot (`windows`, `nvtier.rs:1420-1424`), so a fill is 3 syscalls (`nvtier.rs:2354-2373`) into an
anonymous arena (`AnonMap::new`, `nvtier.rs:1455`). Qwen3.8: about 1.045 MB a part, 3.134 MB a fill (`tierengine.md:25`,
`:350-351`). The box's block layer splits a pread at `max_sectors_kb=1280`, so one part is one NVMe command, and a 4 MiB
pread is about 4 commands in flight from one thread: 6.12 GB/s (`docs/cards/qdsweep.card:17-20`) [measured].

**Counters already there.** `NvTierStats.step_fills` and `step_hits` (`nvtier.rs:1005-1006`). `fill_ns` is a fill
call's wall (`count_fill`, `nvtier.rs:2380-2414`). Each reader kind has `jobs`, `units`, `waits`, `wait_ns` and
`run_ns` (`reader.rs:153-167`): `step.wait_ns` is the wall the `Cols` path waited after its overlap. A `Chain::Step`
fill never reaches the reader, so its wait is only in `fill_ns`. No counter gives L (the miss layers) or prefetch
outcomes.

**Testbeds on the tier.** Qwen3.8 (qwen4exp) builds its tier in `body38.rs:1547-1553`. qwen35moe files go through
`model::host_run` (`crates/gpu/src/arch/qwen3moe/placed.rs:551-560`; `crates/gpu/src/model.rs:1805-1821`): Qwen3.6,
and MiniCPM-V 4.7 text, which is a `qwen35moe` file on the `qwen3` seat (`docs/models.md:13`). Their plan pages
experts through `expert_nvme_tier` (`crates/model/src/arch/qwen35moe/place.rs:624`). The levers that force a paged
shape are `BLOOMERY_CARD_BUDGET`, `BLOOMERY_HOST_ROOM` and `BLOOMERY_NVTIER_BYTES` (`crates/levers/src/registry.rs:250`,
`:43-44`).

### Anchors used below

| Term | Value | Tag |
|---|---|---|
| A6000 paged shape "A": Qwen3.8, plan a, `HOST_ROOM=27G` + `NVTIER_BYTES=20G`, residency off, draft off, prose 512, n 1024 | 42.21 tok/s = 23.7 ms a token (main post-refill) | [measured, `docs/cards/refill-decode.card:100-102`] |
| shape A: fills / evictions in the run (prompt + decode) | 10,473 / 3,705; f not readable on that tree | [measured, `:103-104`, `:111-112`] |
| shape A: f | 3–5 a token, central 4 | [derived, replay, `refill-decode.card:17-28`] |
| shape A: arena tier picks | 191–198 a token over 48 layers (4.0–4.1 a layer) | [derived, replay, same lines] |
| gate plan "G": f | 45.7 a token | [measured untimed, `tierengine.md:465`] |
| one fill at QD 1 (Qwen3.8) | 0.81 ms (fill_ns over fills, mixed kinds, post-refill) to 1.06 ms (gate decode fills, 2.96 GB/s) | [measured, `refill-decode.card:108`; `qdsweep.card:21-23`] |
| drive R(QD) box | 2.51 / 4.15 / 5.67 / 6.48 / 6.77 GB/s at QD 1/2/4/8/16 | [measured, `tierengine.md:467`] |
| one fill at QD 3 | 3.134 MB / ~4.9 GB/s = 0.64 ms (0.63–0.75) | [derived, `tierengine.md:469`] |
| host compute, one Qwen3.8 pick | 23.6 µs | [measured, `tierengine.md:464`] |
| ruler | sd 2.6 %; ±2.44 % at 10 rounds | [measured, `refill-decode.card:74-77`, `:49-50`] |

Derived from these: shape A's layer time is 23.7 / 48 = 0.49 ms. L = 48 (1 − e^(−f/48)) is 3.5 at f 4. The fill wall
is L × t_fill = 3.5 × (0.81–1.06) = 2.8–3.7 ms, 12–16 % of the token [derived]. Removing every fill from the chain is
therefore worth at most +13..+19 % tok/s at shape A [derived].

## 2. Candidates, ranked by expected decode gain over cost

All candidates are host-only (`just ptx-scan` identical). Each changes only *when* bytes are read and the order
independent expert products are computed. The final sum stays in list order, so the bits stay identical (§3.4).

| Rank | Candidate (LES idea) | Shape A gain [derived] | Gate plan G gain [derived] | Cost | Break-even / open condition |
|---|---|---|---|---|---|
| 1 | **C2: part units**, a fill's 3 parts as 3 reader units, so one miss runs at QD 3 (LES "fewer read calls" / "weight layout"; tierengine S2's remaining half) | L × (t_QD1 − t_QD3) = 3.5 × (0.81–1.06 − 0.64–0.75) = 0.2–1.5 ms: **+1..+7 %**, central +4 % | 29.5 × (0.31–0.42) = 9–12 ms: **+11..+17 %** of ~75–85 ms [assumption: G's post-S1 token] | small: `read_pick` split into parts, `Job.units = 3 × picks`, publish at the last part | none: the bytes are the same. Gain > 0 while R(3) > R(1). Opened either way; T2's census reads t_QD1 for the Step kind first |
| 2 | **C1: resident-first on the one-column step**: the `Chain::Step` miss path takes the existing `serve_step_split` (LES "compute resident experts first") | hits per miss layer ≈ 3 × 23.6 µs = 0.07 ms; × L 3.5 = 0.25 ms: **+1 %** (inside the ruler) | ≈ 9 hits × 23.6 µs = 0.21 ms × 29.5 = 6.2 ms: **+7..+9 %** | small: the code exists; the step needs union slabs (`HostRun::prepare_union`, `run.rs:132`) and an allocation-free miss issue (§5 Q6) | gain = L × min(t_hits, t_fill) > the handoff of a reader job (0.05–0.15 ms a job [assumption, `tierengine.md:494`]). At shape A it is near even, so it lands as structural, the prerequisite for C3 (the step must be able to wait on a fill in flight), and is A/B'd at G |
| 3 | **C3: cross-layer prefetch** (LES "cross-layer routing prediction"): at layer l's service, predict l+d's misses and queue them as `Kind::Prefetch` (rank 3, `Claim::Transient`); a step claim on an in-flight prefetch waits on it | r × L × h, h = min(window d × 0.49 ms, residual fill 0.57–0.68): d = 1 gives r × 3.5 × 0.49; **r 0.6 → +4.5 %, oracle r 1 → +7.8 %** | window ≥ 0.77 ms ≥ fill: r × 29.5 × 0.5 ≈ 15 r ms; **r 0.6 → +12 %** | medium: a fourth kind, joining a FILLING slot, cancellation, counters, a predictor | see the break-even below; open only if the replay gives r_miss ≥ 0.35 at d = 1 (shape A) and the oracle ceiling run clears the ruler |
| 4 | C2b: sub-part tiles (part/3, 9 units a fill) | −0.16 ms a fill × 3.5 = 0.56 ms: +2 % | −7 ms (−8 %) (`tierengine.md:357`) | small after C2 | only if C2's measured QD-3 fill is ≥ 0.6 ms (S11's existing gate) |
| — | Contiguous repack (one 3.1 MB pread an expert, LES "weight layout") | 2 syscalls × ~10 µs [assumption] × f 4 = 0.08 ms: **~0** once C2 lands. The block layer already gives QD ~3–4 from one thread (`qdsweep.card:17-20`) and C2 gives the same QD with no file rewrite | ~0 | large: a sidecar of the paged stacks (2× disk) or a rewritten file | do not open |
| — | Direct writes to GPU-visible memory | 0 on discrete: the CPU computes the paged experts (`nvtier.rs:1-9`) | 0 | — | GB10 only: that is S7 + the card leg, the contributor's (`tierengine.md:499`; `docs/contrib/dgx-spark.md` §8). Not this plan |
| — | Layer-wise mixed precision (LES stage 2) | n/a | n/a | a new quant file | not lossless against the shipped file; maintainer question §5 Q3 |
| — | Speculative decode (LES stage 3) | exists: MTP (Qwen3.8), DSpark (Qwen3.6); its `Cols` steps already take C1's path | — | — | read beside as a row (§3), no work |

**C3 break-even [derived].** Let r be the miss recall: the predicted ids that would have missed, over all step
misses. Let q be the precision over predicted *non-resident* ids (only those issue reads; a predicted resident id is
a hint-only claim). Let μ be the probability that a slot a wasted prefetch evicts is requested again before it would
have been evicted anyway. The replay measures μ. A `Transient` claim enters at the cold end (`moe.rs:2157-2161`), so
it takes the LRU-coldest slot and is the next victim itself.

- Gain a token: r · f · h.
- Wasted reads: w = r · f · (1 − q)/q.
- Cost a token: w · μ · t_fill, plus drive contention, which is 0 while the drive is under half the token.

Open C3 when both conditions hold:

\[ \frac{q}{1-q} > \mu \cdot \frac{t_{fill}}{h} \qquad\text{and}\qquad (f + w)\cdot 3.134\,\text{MB} / R(16) \le 0.5\,T. \]

At shape A (h 0.49, t 0.64, f 4, T 23.7) with μ ≤ 0.2 [assumption until the replay], the first condition is q > 0.21.
The second is w ≤ 21 a token, so precision is not binding. To clear the ruler at 10 rounds (≥ 0.58 ms), the bar is
r ≥ 0.58 / (3.5 × 0.49) = **0.34**. MiniCPM's layers are thinner (§3.1): there d = 1 hides only part of a fill, so
the replay also scores d = 2.

**Predictor choice, cheapest first** (`tierengine.md:130-176` names (1) and (3)):

- (P1) The previous token's ids at the same layer predict no miss: they are resident by construction
  (`tierengine.md:132-134`).
- (P2) A **co-activation table**: P(id at l+1 | ids at l). It is learned online from the step's existing tally walk
  (`step.rs:1015`), costs ~0 host time, needs no card change, and can be scored on a Mac from route traces.
- (P3) A **host-side router lookahead**: router_{l+1} applied on the host to layer l's handoff activation `self.x`
  (`step.rs:1027`), on a side thread. It needs no card change. Its cost is one router GEMV a layer: Qwen3.8 512 × 2048
  f32 is about 4 MB, about 0.4 ms on one core at ~10 GB/s [assumption on type and rate], which is about the shape-A
  window. So it is viable only on MiniCPM/Qwen3.6, at 256 × 2048 f32 = 2 MB [assumption, T0 reads it].
- (P4) tierengine's card-side router on the residual: an add-class card change. It is not opened unless P2 and P3
  fail and the oracle ceiling is large.

## 3. Experiment design

### 3.1 Testbeds and rig

| Id | Model, shape | Why | Runner |
|---|---|---|---|
| A | Qwen3.8 UD-Q4_K_XL, A6000 plan a, `HOST_ROOM=27G NVTIER_BYTES=20G`, residency off, `BLOOMERY_DRAFT=off`, prose 512, n 1024, ctx 3072 | refill-decode.card's shape: the one paged tier shape with an admissible number on main (42.21 tok/s) | `just depth-gpu-qwen4exp` (as in `refill-decode.card:61-64`) |
| G | Qwen3.8, gate plan (`BLOOMERY_GEN_PLACE=gate`), draft off, n 1024 | f ≈ 46: the fill-bound regime; C1's only admissible shape | same, `BLOOMERY_GEN_PLACE=gate` |
| M | MiniCPM-V 4.7 35B-A3B Q4_K_M text, A6000, `BLOOMERY_CARD_BUDGET` low enough that every routed expert is on the host, `HOST_ROOM` / `NVTIER_BYTES` so the arena holds about half the routed bytes ("LES-like": most experts behind the drive) | LES's own model; checks that the levers carry across families | `just depth-gpu-qwen35moe` with `BLOOMERY_REF_MODEL` → the MiniCPM file (§5 Q2 on its corpus ids) |
| A+d, M+d | A and M with the draft on (MTP / DSpark) | reads beside: a `Cols` chain already takes C1's path | same |

MiniCPM's expert size: Qwen3.6-35B-A3B's shape is 40 layers, 256 experts, 8 used, hidden 2048, expert ff 512 [all
assumption, T0 reads the header]. At Q4_K_M with gate/up Q4_K and down Q6_K, a fill is 0.59 + 0.59 + 0.86 = 2.04 MB,
and the routed bytes are 40 × 256 × 2.04 MB = 20.9 GB, against the 21.2 GB file (`docs/models.md:35`) [derived]. A
fill takes 0.81 ms at QD 1 and about 0.45 ms at QD 3 [derived]. A layer takes about 0.25 ms [assumption: card ≈ 5 ms
a token (`README.md:50`, 193.4 tok/s whole-card) plus host 320 picks × ~15 µs (23.6 µs scaled by bytes)], so prefetch
depth d = 2 is needed to hide a fill. The M budgets are chosen by T2's census so that f ≥ 5 a token, which keeps every
lever's predicted effect above the ruler.

### 3.2 Counters to add (one `Kind` change, `just records-refresh`)

Keys go on the `nvtier` record (`crates/gpu-gates/src/record.rs:1783`, `nvtier_of` `:3885`), per span (prompt, steps):

- `step_miss_layers`: L directly, counted in `fill_missing`/`prefetch` when picks are non-empty under `Claim::Step`.
- `step_fill_wall_ns`: the step chain's fill wall on the caller, either chain. The pool path is not in
  `KindStats` today.
- `step_hits_under_read_ns`: the hits group's compute while the ticket was pending (`serve_step_split`'s g = 0 span).
- `prefetch` kind in `ReaderStats` (`jobs`, `units`, `waits`, `wait_ns`, `run_ns`), plus `prefetch_issued`,
  `prefetch_resident` (the predicted id was already filled), `prefetch_hit` (a step claim found it filled by a
  prefetch), `prefetch_late` (a step claim joined it while FILLING; its join wait is in `prefetch.wait_ns`),
  `prefetch_wasted` (evicted or a token passed with no step claim), `prefetch_cancelled` (dropped from the queue
  before a unit ran).
- `induced_misses`: a step miss on an id whose last eviction was by a prefetch claim (one tag byte an id, set at the
  eviction, cleared at the fill).
- Probe only: `lookahead_picks`, `lookahead_hits`, `lookahead_miss_hits` by predictor, for r and q in the engine.

Identities the gate pins (`gate-gpu-nvtier`): `prefetch_issued == hit + late + wasted + cancelled` at the span's end
(an open prefetch counts as wasted at the span's close), and `step.units + prompt.units + victim.units +
prefetch.units == 3 × fills_through_reader` after C2.

### 3.3 Baseline, ablation order, pass/fail

The baseline is main at the plan's start, each shape's own arm in the same lease. The ablation is cumulative, each arm
a tree: base → +C2 → +C1 → +C3(oracle) → +C3(predictor). Each A/B compares a tree with the previous tree, so each
lever reads its own term. Each is a `docs/cards/<slug>.card` with the band below, `rounds: 10`, `sd: 2.6 %`, and the
refill card's conditions: one lease, `BLOOMERY_AB_LOAD=arm`, warm-up discarded, rotation, `tokens-differ rounds: 0`,
and equal plan and nvtier budget fields.

| Step | Kind | Shape | Predicted | Pass (decide-in) | Fail (finding) |
|---|---|---|---|---|---|
| E0 replay (Mac, no card) | functional | traces of A, G, M | r_miss, q, μ for P2 at d = 1, 2 | r ≥ 0.34 (A) → C3 opens with P2 | r < 0.34: P3 probe (E3b); both < 0.34: C3 closes, C1+C2 stand |
| E1 census (box, no lease) | functional | A, G, M, A+d | f, L, step t_fill, `step.wait_ns`, by chain | fixes M's budgets (f ≥ 5) and each card's band | f at A < 3: shape A cannot show C1/C3; G and M carry them |
| E2 +C2 | ab | A, G | A +1..+7 %; G +11..+17 % | inside the band; the Step kind's run_ns per fill falls to 0.63–0.75 ms | t_QD3 ≥ t_QD1: the drive's QD-3 rate is the wrong term; recheck qdsweep's curve |
| E3 +C1 | ab | G (A read beside) | G +7..+9 %; A +1 % (inside the ruler, no claim) | G in band; `step_hits_under_read_ns` ≈ L × 0.21 ms | a loss at A beyond −1 %: the handoff term (job alloc, wake) is larger than 0.15 ms |
| E3b P3 probe | functional | M | r for router-on-x_l | r ≥ 0.34 → P3 replaces P2 | — |
| E4 +C3 oracle | ab | A, M | the ceiling: A +7.8 %, M from E1 | the ceiling clears the ruler → build the predictor | ceiling inside the ruler: C3 closes before the predictor |
| E5 +C3 predictor | ab | A, M | r × ceiling (E0's r) | inside the band; `induced_misses` ≤ μ · w as the replay said | below: read r and q from the probe keys, then which term (r, h, μ) missed |

The **oracle arm** feeds the true ids of layer l+d from a route trace of the same greedy run as its predictions.
Greedy decoding on identical bits gives identical routing, which the trace's ids check. It is a probe input of the gate
bin (§4 T6), not a lever. Its result is C3's ceiling at that shape, so the predictor is built only when that ceiling
clears the ruler.

Every A/B reads beside, not judged: pp tok/s at P 512 (no change expected: prompt fills sit under the drive), and the
same arms with the draft on.

### 3.4 Bit identity (lossless)

- **Prediction changes no compute.** It only claims and fills slots early. Which experts run and their weights are the
  router's (`step.rs:1006-1024` builds the list from the handoff). The slot bytes are the file's, which
  `NvTier::audit_on` checks on functional runs (`nvtier.rs:2092-2166`).
- **Sum order.** Every path sums in list order. The one-column path uses `serve`; the split path uses
  `sum_batched` (`moe.rs:1937`), whose contract is "column j = `experts_into` of column j bit for bit"
  (`moe.rs:2696-2706`, held by `gate-union`). C1 routes the one-column miss case through that same contract.
- **Gates.**
  - `gate-union` and `gate-ops` (column identity).
  - `gate-gpu-nvtier` (slot bit identity, the identities of §3.2).
  - `gate-gpu-e2e` and `gate-gpu-qwen35moe-e2e` (tokens).
  - `gate-alloc` (no allocation added on a hit-only step; §5 Q6 for the miss path).

  FAIL-first per clause:
  - Sum the split path in landing order: `gate-union` goes red.
  - A prefetch that pins a slot a step needs: the nvtier clause goes red.
  - A step that reads a FILLING slot without joining: refused by name (`TierError::Unfilled`).
- **A/B runs.** `tokens-differ rounds: 0 of 10` is a condition of every card.

## 4. Implementation tasks (one worker each; host-only; files to touch)

| # | Task | Files | Proof / gates | Needs |
|---|---|---|---|---|
| T0 | Read MiniCPM-V 4.7's header: layers, n_expert, n_used, part types and bytes, router tensor type. Write them into §3.1, replacing the [assumption]s | this doc only | — (a Mac reads the GGUF header) | — |
| T1 | Counters of §3.2, except the prefetch ones, plus `step_miss_layers` and `step_fill_wall_ns` | `crates/gpu/src/host/nvtier.rs` (`Stats`, `NvTierStats`, `fill_missing`, `count_fill`), `nvtier/reader.rs` (`KindStats`), `crates/gpu-gates/src/record.rs` (`nvtier` kind), `just records-refresh` | move class, ptx-scan identical; `gate-gpu-nvtier`'s kind test re-pinned; FAIL-first: drop a key → red | — |
| T2 | E1 census: functional runs at A, G, M, A+d; one table into this doc (f, L, t_fill by kind, `step.wait_ns`); pick M's budgets | this doc; logs in rig-log | functional, no lease | T1, a box |
| T3 | E0 replay: an arena-books replay of `pick_slot`/`Claim` rules (`nvtier.rs:283`, `moe.rs:2150-2161`) over route traces, scoring P1/P2 at d = 1, 2 for r, q, μ. Self-test listed per `just check-recipes` | `tools/flow/lookahead_replay.py` (new), `tools/recipes.py` self-test list; traces from `BLOOMERY_ROUTE_TRACE` one-row runs | the self-test on a synthetic trace with a known r | traces from a box (functional), then a Mac |
| T4 | C2: part units. `read_pick` becomes per-part `read_part`; `Job.units = 3 × picks`; a pick publishes when its third part lands; the pool path `fill` chunks by part too | `nvtier.rs` (`PickRead`, `read_pick`, `fill`, `prefetch`, `fill_victims`, `Inflight`), `nvtier/reader.rs` (unit doc) | ptx-scan identical; `gate-gpu-nvtier` (bit identity, units = 3 × fills, a short part refused by name `nvtier.rs:2362-2369`); E2 card | T1 |
| T5 | C1: the `Chain::Step` miss path. In `experts_into_on` the claim becomes `prefetch_pinned`: a ready ticket keeps today's path; a pending ticket goes to the one-list split path. `HostRun` prepares union slabs for one column on a tiered load. The miss issue makes no allocation (pooled `Job`s and picks) | `crates/model/src/moe.rs` (`experts_into_on`, `serve_step_split`), `crates/gpu/src/host/run.rs` (`prepare_union`, `experts_into`), `nvtier.rs` (`Inflight::queue` buffers), `crates/model/tests/alloc.rs` (a miss case for `PoolTier`) | `gate-union`, `gate-ops`, `gate-alloc`, `gate-gpu-nvtier`, `gate-gpu-e2e`, `gate-gpu-qwen35moe-e2e`; FAIL-first: landing-order sum → red; E3 card | T1 |
| T6 | C3 mechanism with an oracle input: `Kind::Prefetch` rank 3 at most 8 of 16 threads while a step runs (`tierengine.md:359`); `Claim::Transient`; a step claim on a FILLING-by-prefetch slot joins its job instead of refusing; queued units cancel when the step claim covers them; prefetch counters and `induced_misses`; the gate bin reads oracle ids from a route trace (`--lookahead-oracle DIR`, a probe argument of `generate_qwen3moe`, not a lever) | `nvtier.rs` (claim/join, `pick_slot` caller), `nvtier/reader.rs` (`Kind`, `KINDS`, cancel), `moe.rs` (`TierSlots::prefetch_hint`, default no-op), `crates/gpu/src/host/step.rs` (`serve_one`: issue for l+d before `experts_into`), `crates/gpu-gates/src/bin/generate_qwen3moe.rs`, `record.rs` | `gate-gpu-nvtier` clauses of §3.2; FAIL-first: a prefetch that pins a step's slot → red, a non-joining step → refused by name; tokens equal under the oracle; E4 card | T4, T5 |
| T7 | Predictor by E0/E3b's verdict: P2 as an online co-activation count fed by the tally (`step.rs:1015`), or P3 on a side thread over `self.x` | `crates/gpu/src/host/step.rs`, a new `crates/gpu/src/host/lookahead.rs` | `gate-gpu-nvtier` probe keys; `gate-alloc` (the table is sized at load); E5 card | T6, E0 ≥ bar |
| T8 | Land each A/B's row in rig-log; strike or confirm this doc's [derived] bands in place; update `tierengine.md` S2/S3/S5 status rows | this doc, `docs/research/tierengine.md` §5 | — | each E |

Order and owners: T0, T1 and T3's script run in parallel (Mac). T2 needs T1 and the box. Then T4, then T5; they share
`nvtier.rs`, so one owner per round, T5 rebased on T4. T6 needs both, and T7 needs T6 and the E0/E3b verdict. Every
task is host-only. `just narrow` applies, and a kernel that moved is a stop.

## 5. Questions for the maintainer (nothing here changes a release, a default or a public API)

1. **Which shape is the 0.3.3 claim?** A (the public paged A6000 shape, draft off), G (the gate plan) or M (MiniCPM,
   LES-like)? C1 is under the ruler at A and admissible only at G. The README's rows run the draft on, where `Cols`
   already has C1.
2. **MiniCPM in the timing runners.** There is no profile under `tools/ref/models/`. Is `qwen35moe.sh` with
   `BLOOMERY_REF_MODEL` acceptable, or does MiniCPM get its own profile? Is its tokenizer equal to Qwen3.6's? If not,
   its `corpus-prose.ids` must be made.
3. **LES stage 2 (mixed precision)** produces a different file and is not bit-identical to the shipped Q4_K_M. Is it
   out of scope, or a separate "file choice" row?
4. **LES stage 3.** The MiniCPM DFlash draft config reads (`crates/hf/src/dspark.rs:9`, fixture
   `crates/hf/tests/fixtures/dspark-minicpm-v-4.7-config.json`). Is it served on the `qwen3` seat in 0.3.3? This plan
   only reads drafted rows beside.
5. **A prefetch lever.** A same-binary arm needs a lever row in `crates/levers/src/registry.rs`, a new runtime lever
   with a default. Without one, every C3 claim is tree-against-tree, which keeps claims under 1 % inadmissible. Should
   T6 add it, and with which default?
6. **Allocation on the miss path.** `4d6dbeb2` states that the `Cols` miss path allocates on issue and that `gate-alloc`
   does not see it. T5 moves the one-column step onto that path. Does T5 make it allocation-free (as planned) before
   landing, or may it land with the `Cols` path's current allocation and a dated note?

[les]: https://github.com/tc-mb/llama.cpp-omni/blob/master/docs/les/lookahead-expert-scheduling.md
