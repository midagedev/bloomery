# Kimi K3 / K3.1: a prompt pass for the placed body (`--place bp`)

A derivation and design note (2026-11-06, round `k3fill`; no code changed, no timed run). It prices the step a placed K3
load runs today, a pass of R rows run layer by layer, and how the pass is cut between the common owners and K3's own
code. Every number is cited or marked `[derived]` with its terms. Nothing here is a measurement unless it says so.

## 0. Result

| Quantity | Today (one id a step) | Pass of R rows | Basis |
|---|---|---|---|
| decode, K3 UD-IQ1_S, A6000 stage + 3090 tier + host + NVMe | 0.74–1.4 s a step, **≈ 1.3 tok/s** (band 0.7–1.6) | unchanged (the step is not touched) | §2 `[derived]` |
| pp @ P = 512 | ≈ 0.77 s a row, **≈ 1.3 tok/s** | R = 512, one pass ≈ 45 s, **≈ 11 tok/s** (band 5.5–12) | §3 `[derived]` |
| pp @ P = 4096 (K3 only; a K3.1 file refuses ctx > 2048) | ≈ 1.3 tok/s | 8 passes of 512, **≈ 11 tok/s** (band 5.5–12); one tail pass of 4096 is host-bound, 15–40 tok/s, later | §3 `[derived]` |
| `--score`, 150 rows × (≈ 100 prompt + 128 teacher) | 150 × 227 steps ≈ **7 h** (band 5–9 h) | 150 passes of 228 rows ≈ **1.8 h** (band 1.7–3.3 h) | §4 `[derived]` |

The pass is NVMe-bound at every R that matters: a pass of R ≥ 512 reads each paged expert once (275.6 GB on this
plan) where R steps read 4.9 GB each. **The one term derivation cannot close is the paged read path's bytes and rate
under a pass**: this plan reserves no NVMe arena (`arena=0`), so the paged experts are read through the file mapping's
page cache, where a layer-ordered scan of 521 GB of host experts through ~250 GB of RAM may evict the host share it
should keep (§2.3). Expected value at the arena's O_DIRECT path: 275.6 GB at 6.77 GB/s = 40.7 s a pass of 512; the
page-cache bound is 275.6–521.5 GB at 4.6–6.8 GB/s, 41–113 s.

Design (§5): a K3 walk of `Overlap { units: 1, cols: R, port: Batch }` through the common batch port, its mixers run
in chunks of the existing 8-row pass (each row the step's bits), its routed block over all R rows at once (one host
union call a layer: the read-once gain). No kernel is added; the class is **add** with bit-identical rows (§6).
Decisions for the lead are in §8.

## 1. The facts the derivation reads

**The file** (K3 UD-IQ1_S, `/models/Kimi-K3-UD-IQ1_S`, 594,040,923,616 B on disk; header read 2026-11-06 by a
read-only script over every shard's tensor table):

| Part | Bytes | Note |
|---|---|---|
| routed experts | 545,349,697,536 (254 IQ1_S + 22 IQ2_XXS stacks) | 92 routed layers × 896 experts = 82,432; **6.616 MB an expert** (3 × 3584 × 3072 weights) |
| shared experts (`shexp`, Q6_K) | 9,970,974,720 | 2 a layer, full width |
| KDA projections (q, k, v, g, o; Q6_K) | 26.7 GB | 69 delta layers |
| MLA, router (F32), latent pair, dense layer 0, norms | ≈ 10.0 GB | |
| `token_embd`, `output` (Q6_K) | 0.96 GB each | both read on the host on this plan |

Dims (header): `n_embd` 7168, 93 layers (69 KDA, 24 MLA: 4, 8, …, 92, 93), 896 experts, 16 used, 2 shared,
`expert_latent_length` 3584 (the experts' width), `expert_feed_forward_length` 3072, `attn_res.block_size` 12
(8 banks), vocabulary 163,840. One expert row (one token through one expert) is 3 × 3584 × 3072 = **33.03 M MAC**.

**The plan** (`generate_kimi_k3 --plan --place bp` on that file, box, 2026-11-06, round `kimireal`,
`~/.cache/bloomery/specs/kimireal/decode.txt`): stage A6000 (head on the host: the trunk with the head is 648.6 MB past
the card's free bytes), tier 3090; `card_experts=25 (162,656,256 B)`, `host_experts=78832`, of which
**245,952,344,064 B held on the host** and **`paged=275581243392`** through the NVMe tier, `arena=0 arena_slots=0`,
`room=261832790016`, `need … cache=261295419392 locked=0`. The tier holds 82,432 − 78,832 − 25 = 3,575 experts
(23.65 GB) `[derived]`.

| Tier | Experts | Share | Bytes a decode step reads (16 × 92 = 1,472 expert reads) |
|---|---|---|---|
| stage card | 25 | 0.03 % | 0 |
| 3090 tier | 3,575 | 4.34 % | 0.42 GB `[derived]` |
| host RAM | 37,177 | 45.10 % | 4.39 GB `[derived]` |
| NVMe paged | 41,655 | 50.53 % | 4.92 GB `[derived]` |

Uniform routing is assumed throughout; it maximises a pass's union (a skewed router touches fewer experts in a pass and
more of the host share in a step), so the pass's numbers are its worst case.

**The box** (rig-log `docs/bloomery/facts.md`, `log/2026-10-11.md`): Threadripper PRO 5975WX, 32 cores, AVX2/FMA,
no AVX-512; memory read 147.7 GB/s (provisional); 251 GiB RAM; NVMe O_DIRECT read **6.77 GB/s at QD 16** (knee at 8)
on nvme0 (`/models`); A6000 dense reads ~700 GB/s (the contract's figure for a kernel near the bound); both cards
PCIe 4.0 × 16.

**The K3.1 quant** (tight map, ~580 GB, lands tonight) has the same experts (545.3 GB) and a Q4_K trunk of
≈ 33 GB `[derived: the Q6_K trunk × 4.5 / 6.5625 plus the F32 parts]`: the stage card then holds the head and
≈ 16 GB of experts (≈ 2,400, 2.9 %), and the paged share falls to ≈ 260 GB `[derived]`. Every row below moves by
under 6 % for it; the tables are the K3 file's.

**The k3shape set** (`/home/user/stage/k3shape`, 1,561,252,818,400 B): experts MXFP4 1.446 TB (17.55 MB an expert),
trunk BF16 ≈ 112 GB, which the load widens to f32. Its `--plan --place bp` is expected to refuse at the stage card's
dense floor (≈ 110 GB of trunk past a 50.7 GB card) `[derived, not run]`: a cold `cargo oxide` build on a box whose 57
cores run `llama-quantize` was not worth that refusal. The set is the quantizer's source, not a placed load; the plan
to read is the IQ1_S quant's when it lands.

## 2. The step today

### 2.1 Terms

| Resource | Per step | Wall |
|---|---|---|
| A6000, trunk weights (46.8 GB at 1 row) | 46.8 GB | 67 ms `[derived, 700 GB/s]` |
| 3090 tier slots | 0.42 GB | 0.5 ms, under the host leg |
| host DRAM, held experts | 4.39 GB | ≥ 30 ms at 147.7 GB/s |
| host compute, 1,407 expert rows | 46.5 G MAC at one column a weight row | 0.1–0.2 s `[derived: an IQ1_S 1-column row is decode-bound; band from 0.25–0.5 TMAC/s]` |
| NVMe, paged experts | 4.92 GB (plan split) … 9.3 GB (page cache holding nothing of the host share) | 0.73–1.4 s at 6.77 GB/s |
| PCIe | 92 × (a latent row down, a sum up) ≈ 2.6 MB | ~0 |

### 2.2 Timeline

Per layer the card's front (mixer, router, latent down) runs, the host serves the layer's 16 slots — the tier's
1 under it — and the card's back waits. The NVMe read of a layer's ~8 paged experts cannot start before that layer's
router, so nothing hides it: the step is the sum over layers of `front + max(host, shadow) + back`, and the host term is
the NVMe read, ≈ 8 ms a layer of 53 MB. Step ≈ 92 × (8.0 + ~0.4) ms ≈ **0.77 s**, ≈ 1.3 tok/s `[derived]`, the band
0.7–1.6 tok/s spanning the page-cache share (§2.3). The kimireal runner walls are consistent with it (not timed
numbers: `gpu-gate.sh` wall less `load_s`, 18 s over ~14 steps and 142 s over ~220 steps, 0.64–1.3 s a step). A
shorter card term moves nothing: it sits beside an NVMe term ten times its size.

### 2.3 The read path at `arena=0`

`NvTier::of_paged` (`crates/gpu/src/host/nvtier.rs:2082`) returns `None` when the arena grant is 0, so this load has
no NVMe tier object: the host union reads every expert through the file mapping. The plan's "held on the host" share is
page cache (`cache=261295419392 locked=0`), not pinned memory, so the kernel's LRU decides what stays. Under a step's
random access the hit rate is about the cache's share of the host experts (≈ 47 %), which is the plan's split. Under a
pass, every host expert is touched once per pass in layer order, a cyclic scan of 521.5 GB through ~250 GB: a page's
second touch comes after more than the RAM's worth of other reads, so the refault distance never activates it and the
scan can evict the share a step needs. A pass then reads between 275.6 GB (the split holds) and 521.5 GB (nothing
held). The common arena path avoids it: the paged ids are read `O_DIRECT` into the arena in batches of
`UNION_BATCH_IDS` (16) with the fill reader ahead of the compute, and the host share's pages are left alone. Its floor
on this plan is `max(16 × 1 + 5, 2 × 16 + 5)` = 37 slots × (92 paged layers × 6.616 MB) = **22.5 GB**
`[derived from crates/placement/src/placement/room.rs:458-466, 557-563]`, taken from the host share (paged
275.6 → 298 GB, +8 %). Why this plan reserved none is to read in stage 2 (`HostTotals::nvme_arena_bytes`, the split
dial, `BLOOMERY_NVTIER_BYTES`).

## 3. A pass of R rows

### 3.1 The union

Layer-major, a layer's routed block sees R rows at once; the host serves the union of their experts once. Expected
distinct experts a layer: U(R) = 896 · (1 − (1 − 16/896)^R) `[derived, uniform]`.

| R | U(R) | NVMe a pass (paged share, split holds) | at 6.77 GB/s | R steps' NVMe | host MAC | host at 0.7 / 1.4 / 2.0 TMAC/s |
|---|---|---|---|---|---|---|
| 8 | 120.3 | 37.0 GB | 5.5 s | 5.8 s | 0.37 T | 0.5 / 0.3 / 0.2 s |
| 64 | 613.2 | 188.6 GB | 27.9 s | 46.5 s | 3.0 T | 4.3 / 2.1 / 1.5 s |
| 128 | 806.7 | 248.1 GB | 36.7 s | 93.0 s | 6.0 T | 8.5 / 4.3 / 3.0 s |
| 228 | 881.3 | 271.1 GB | 40.0 s | 165.7 s | 10.6 T | 15.1 / 7.6 / 5.3 s |
| 256 | 887.1 | 272.8 GB | 40.3 s | 186.1 s | 11.9 T | 17.0 / 8.5 / 6.0 s |
| 512 | 895.9 | 275.6 GB | 40.7 s | 372.2 s | 23.8 T | 34.0 / 17.0 / 11.9 s |
| 1024 | 896 | 275.6 GB | 40.7 s | 744.3 s | 47.6 T | 68 / 34 / 24 s |
| 4096 | 896 | 275.6 GB | 40.7 s | 2977 s | 190.5 T | 272 / 136 / 95 s |

R = 8 (today's card-alone pass) saves 6 %; R = 256 4.6×; R = 512 9.1×. Past R ≈ 512 the NVMe term no longer grows
and the host's MACs do: the crossing is R* = 40.7 s × rate / 46.5 G MAC ≈ 610–1,750 rows. The host rate is the IQ1_S /
IQ2_XXS wide tile's (`qdot::WIDE_COLS` = 32, `TILE_COLS` = 8) at ~R/56 columns an expert (9 at R = 512); no record
measures it, so it carries the 0.7–2.0 TMAC/s band `[derived: 7.4 TMAC/s int8 peak of 32 Zen 3 cores, 10–27 %]`.

### 3.2 The card at R rows

Each projection runs in chunks of ≤ 8 columns (`site::kgemv` caps at `KGEMV_COLS`; the rows kit is the step's bits per
row), so the trunk is read once a chunk: R = 512 is 64 × 46.8 GB = 3.0 TB at 700 GB/s = 4.3 s a pass, of which the
shared experts' 21 % runs in the host leg's shadow. The KDA recurrence runs in position order inside
`kda_delta_lanes`, one launch a chunk; at ~10 µs a token-layer it is 512 × 69 × 10 µs = 0.35 s `[derived, the per
token term is unmeasured]`. MLA at R rows attends each row to its own prefix: at depth ≤ 4096, 24 layers × 4096 rows ×
96 heads × 1,088 × 2,048 average keys × 2 flop = 42 TFLOP over the whole prompt, ~4 s at 10 TFLOP/s effective. The
eager launches (~40 a chunk-layer, 64 × 93 chunk-layers) are ~0.2 M × 4 µs ≈ 1 s of host enqueue, behind the card's
own work. A GEMM over R rows would change every row's sum order for a term the NVMe shadow prices at 0, so it is not
here.

### 3.3 Timeline of a pass (R = 512, one unit)

Per layer: the card front (mixers in 64 chunks, the block's mix and norm, the router, the latent down: ≈ 30 ms), the
download of 512 latent rows (7.3 MB, < 1 ms), the host union (NVMe 275.6 GB / 92 = 3.0 GB a layer, 440 ms at 6.77
GB/s; its MACs 0.13–0.37 s; DRAM 2.7 GB, 18 ms), with the shared stack, the card's slots and the tier's slots in the
shadow (≈ 15 ms), then the upload and the back (≈ 10 ms). The wall is `Σ front + max(host, shadow) + back` ≈
92 × (30 + 440 + 10) ms ≈ **44 s** on the arena path, where the fill reader runs ahead of the union's compute so the
host term is `max(NVMe, MACs)`; on today's page-cache path the faults land on the compute threads, so the host term is
between that max and the sum (0.44–0.81 s a layer), and the bytes are §2.3's band. pp512 ≈ 512 / 44 ≈ **11.6
tok/s** `[derived]`; band 5.5–12 across §2.3's read band. A second unit (`units: 2`) would hide the ≈ 40 ms of card
work a layer under the other unit's host leg: ≤ 9 %, left for later.

P = 4096: 8 passes × ≈ 45 s ≈ 360 s, ≈ 11.4 tok/s. One pass of 4096 through the union's group tail
(`UNION_TAIL_MAX_COLS` = 4096) reads NVMe once (40.7 s) but runs 190.5 T MAC on the host: 95–272 s, 15–43 tok/s. It
needs 8× the card's pass buffers (§5.3) and is the second step, not this one. A K3.1 file is refused past ctx 2048
(`Hparams::check_ctx`), so its P = 4096 row does not exist.

## 4. Scoring (`generate_kimi_k3 --score`)

Today a line runs its ≈ 100 prompt ids and 127 forced ids one step each: 150 × 227 × 0.77 s ≈ 26,200 s ≈ **7.3 h**
(band 5–9 h across the step's band). As one pass of 228 rows a line (the prompt and the completion, the last 128 rows'
logits read): U(228) = 881.3, NVMe 271.1 GB = 40.0 s, host MACs 5–15 s under it, card ≈ 2 s, the host head over 128
rows (1.17 G MAC a row, 150 G MAC, ≈ 0.3 s as one 128-column projection; 84 MB of logits) ≈ **42 s a line, 1.75 h**
(band 1.7–3.3 h). The next lever, outside this round: several lines in one pass on several state lanes (the KDA stores'
lanes, a cache range a line) share one NVMe read; at 4 lines a pass the host MACs (4 × 228 × 46.5 G = 42 T, ≈ 30 s)
meet the NVMe term and the 150 lines take ≈ 30 min `[derived]`.

## 5. Design

### 5.1 What is common, what is K3's

V4.1's prompt batch walker (`crates/gpu-deepseek41/src/prompt_walk.rs`: `Batch`, `BatchBufs`, `BatchAttention`,
`Entering`, the units' host staging) is shared by V4.1, Mistral Large 4, Solar Open 2 and MiniMax M3, but it lives in
`gpu-deepseek41`, which depends on `bloomery-gpu`; K3 lives in `bloomery-gpu` and cannot call it. Its block is
`Routed<R>` (`shared_routed.rs`: one ungated shared expert, the residual association, the `ds41_*` card tiles), which
is not K3's routed block (`routed_ffn` with the latent pair, the routed norm, the SiTU shared stack) either way. What
V4.1 and K3 already do the same way is already common and in `bloomery-gpu`, and the pass uses it unchanged:

- the schedule: `runtime::sched::walk` at `Overlap { units: 1, cols: R, port: PortKind::Batch }`, a `LayerProgram`;
- the host port: `crate::host::BatchLeg` / `BatchPort` (download, the union serve, upload, the tier legs and
  `join_tiered`), `Hybrid::prepare_batch`, `HostRun::prepare_union`;
- the host union (`model::moe::serve_union`, each column's bits the one-column serve's, `moe.rs:1736-1752`) and its
  NVMe arena batches (`serve_tiered`);
- the slot places of m columns (`CardSlotKernels::enqueue_places`, `routed_block::tier_places_rows`), the card leg at
  m tokens (`card_leg::LegRun.m`), the slot sums at m rows (`CardSlotKernels::enqueue_card_acc`, `CardAcc.m`);
- the call cut: `runtime::prompt::{call_end, call_batches}`;
- the plan: `PlanInputs::with_prompt_reserve` (the family owns the formula, the load holds its allocation to it,
  `place_hybrid.rs:15-20`) and the tier leg's batch bytes `workstation::tier_leg_batch_bytes(latent, ff, n_used,
  chunk, rows)`.

K3's own, each forced by an architecture fact: the AttnRes mixes (banks of R rows), the KDA recurrence and the MLA
cache (run through today's `kda::rows` / `mla::rows` in chunks), the latent down/up and the routed norm, the SiTU
shared stack, the IQ1_S / IQ2_XXS experts (already the host run's types).

Two cuts were weighed:

- **(a) lift the walker into `bloomery-gpu`**: `UnitBuf`, `Stage`, the `BatchBufs` core, the walk as a `LayerProgram`
  over `BatchLeg` and a block trait that `Routed<R>` and K3's block both implement, re-exported from
  `gpu-deepseek41` so its four users' imports stand. A move of ≈ 600 lines whose proof is V4.1's, ML4's, Solar 2's and
  MiniMax's prompt gates (the real tier), for a second user whose block shares none of `Routed<R>`'s body.
- **(b) a K3 walk** (`crates/gpu/src/arch/kimi_k3/pass.rs`, ≈ 450 lines): a `LayerProgram<Port = BatchLeg>`, the
  batch analogue of `placed.rs`'s `StepWalk`, over the common parts above. Its one copy of shared logic is the host
  staging of R embedding rows, positions and key counts (≈ 30 lines, `prompt_walk.rs:146-176` and `:708-732`).

**Recommendation: (b) now**, and (a) named for the lead to schedule once a second `bloomery-gpu` family wants a
batch walk; the staging copy is the piece (a) would remove.

### 5.2 The walk

`Body::prompt` on a placed load cuts the ids into passes of ≤ R (`call_batches`), and each pass is:

1. stage: the R ids' embedding rows from the host table (`EmbedHost`), their positions and live key counts, one copy
   each to the card;
2. `sched::walk((1, R, Batch), 93, leg, walk)`; at layer l:
   - **front**: the mixer's input mix over the banks (`attn_res`, one block a row, banks `R · n` apart), the bank copy
     on a banking layer, the mixer **in chunks of ≤ 8 rows** in position order through today's `kda::rows` /
     `mla::rows` (each chunk the 8-row pass's launches), the sum after it; then the block's mix and `ffn_norm` over R
     rows; on a hosted layer the router over R rows, the latent down (8-column chunks) into the port's activation, the
     card and tier places of the R × 16 picks, and the download (`BatchLeg`, tiered where the tier holds experts of
     the layer); a whole-card layer (the dense layer 0, a layer the card holds whole) runs its block over the R rows
     and returns;
   - **shadow**: the shared stack over R rows (8-column chunks), the card's slots through the card leg at m = 8
     chunks, summed per row in slot order;
   - **back**: the port's upload of the host sums, the tier rows joined (`join_tiered`), per row
     `(card + tier) + host` (the step's association, `placed.rs` `back`), the routed norm, the latent up, the update
     added;
3. the tail: no head (`Open`), the last row's head (`Last`, the decode's), or the last N rows' heads (`Rows(N)`, the
   score's): the final mix of those rows, read on the host by the host head (`Head::reads_on_host`) as one N-column
   projection.

The mixers' chunks need no new store: the KDA conv ring already holds 3 + 8 rows a pass (`place.rs` test
`the_layers_stores_are_the_headers_widths`), and a chunk of 8 is today's pass, so the plan's store bytes do not move.

### 5.3 Buffers at R rows

Only what crosses layers or the host boundary is R rows wide; the mixers' intermediates stay at today's 8-row
`Pass` scratch. Per row `[derived]`:

| Buffer | Values a row | Bytes a row |
|---|---|---|
| the sum, both halves | 2 × 7168 | 57,344 |
| the banks (8 slots) | 8 × 7168 | 229,376 |
| the block's normed input, the shared stack's output, the update | 3 × 7168 | 86,016 |
| the latent activation, the card sum, the tier sum, the host sum, the routed sum | 5 × 3584 | 71,680 |
| ids, weights, card and tier places | 4 × 16 words | 256 |
| **total** | | **444,672** |

R = 256: 113.8 MB; R = 512: 227.7 MB; plus the 8-row pass's scratch `[derived]`. ~~≈ 15–25 MB at ctx 4096~~: the
latent mixers' attention partials alone are `q rows × segments × 512` f32, 8 rows × 96 heads × 80 segments × 512 × 4 B
= **125.8 MB** at ctx 4096 (one row: 15.7 MB) `[derived from flash_gqa's SEGMENTS and LATENT]`, so the inner latent
run is a plan choice too (`PassShape { rows, mla_rows }`, mla_rows 8 or 1; §10). The plan
reserves it (`with_prompt_reserve`), and the reserve displaces card experts first: on the K3 file the stage card's 25
experts (162.7 MB) cover R = 256 (≈ 135 MB with the 8-row scratch); R = 512 (≈ 250 MB) is ≈ 85 MB past them — past
the 25 the A6000 has no experts left to give, so R = 512 on the K3 file needs the plan's margin or the head-on-host bytes reread
(stage 2 reads the plan's refusal); on the K3.1 quant (≈ 16 GB of card experts) R = 512 displaces ≈ 36 of ≈ 2,400. The
3090 tier's batch bytes grow to `tier_leg_batch_bytes(3584, 3072, 16, chunk, R)` (≈ 0.1–0.25 GB at R = 512,
displacing ≈ 15–40 of its 3,575 experts); the host's pinned port sets (2 × R × (2 × 3584 + 32) × 4 B = 29 MB) and
union slabs (`UnionScratch::new_routed(3584, 3072, R, 16)`, ≈ 0.3 GB at R = 512) come out of the host room. Each of
these moves a clause of `gate_kimi_k3_place` (the plan's card and tier counts), which stage 2 re-derives.

### 5.4 Where the pass lives

The pass's buffers are made at load beside `Placed` when the plan reserved them, counted in `refuse_scratch_past` /
`refuse_reserve_past`, and held by `Placed` so `Body::prompt` (and the serve seat's prefill, which calls it) runs the
pass without a caller-owned `Pass`. `Pass::new`'s refusal of a placed body stays for the card-alone `Pass` (a placed
load runs the placed pass), its message updated.

## 6. Order and class

No launch sums across rows, and each row's sums keep the step's order:

- the mixers: chunks of today's pass, each row the step's bits (`gate-gpu-kimik3-pp`'s `rows_are_steps`);
- the AttnRes mix: one block a row, the banks at stride `R · n`;
- every projection: ≤ 8 columns through the rows kit, each column the one-column launch's bits;
- the host union: each column's sum in its list order from zero, the one-column serve's (`moe.rs:1736-1752`);
- the card and tier slots: per (row, slot) launches and the slot-order sum per row;
- the back: `(card + tier) + host` per row, as the step.

So every row of a pass is the step's bits, the stores advance as R steps would, and the change is the **add** class with
no new kernel (`just ptx-compare BASE generate_kimi_k3` = identical) and a new engine caller (`Body::prompt`). No error
model is needed; the e2e count pin does not move.

## 7. Gate plan (stage 2)

As built (§10) the clauses are `gate_kimi_k3_place`'s `pass state`, `pass batches`, `pass score`, `pass refused` and
`pass cli`; the plan below is the stage-1 proposal they came from.

On the K3 fixture, placed as `gate_kimi_k3_place` places it (a budget that splits the experts over the stage card, the
tier and the host; an arm whose room pages experts through the NVMe tier):

- `placed_pass_rows_are_steps`: a prompt of 37 ids (four chunks and a ragged one) as one pass with `Tail::Rows(37)`
  against the same load stepped id by id: `differing_rows = 0` over the logits' bits, for R ∈ {1, 8, 37} and a cut of
  37 into passes of 16;
- `placed_pass_state_advances_as_steps`: after the pass, 4 steps' bits equal the stepped load's;
- `placed_pass_reads_each_expert_once`: the port's `UnionCols.experts` for each hosted layer equals the distinct ids
  the pass's rows routed (`BatchPort::routed_ids`), and the NVMe tier's union claims (`NvTier` stats) one claim an id a
  layer;
- `placed_pass_refused`: a pass past the reserve's rows, a pass at a position the stores do not stand at, a `Rows(N)`
  of more rows than the pass, a placed load whose plan reserved no pass bytes, each by name;
- `score_pass_is_score_steps`: `generate_kimi_k3 --score` on a fixture line gives the same JSON lines as the stepped
  score (`--score-walk steps|pass`, a bin flag, not a lever).

FAIL-first mutants, each red on its named clause before the fix: the back's association turned `card + (tier +
host)`; the banks' stride left at 8 rows; the download of the first chunk only; the pass leaving `held` behind; the
union served per chunk (each expert read once a chunk: `reads_each_expert_once` red). Owning gates beside them:
`gate-gpu-kimik3-place`, `gate-gpu-kimik3-pp`, `gate-gpu-kimik3-e2e`, `gate-gpu-kimik3-serve`, `gate-gpu-lib`; the
static tier; `just ptx-compare` for the no-new-kernel claim.

## 8. Decisions for the lead

1. **The cut**: (b) a K3 walk over the common port now; (a) the walker's lift into `bloomery-gpu` later.
2. **R and the reserve**: R = 512 (`UNION_MAX_COLS`; the NVMe term stops growing there) on the K3.1 quant, where it
   displaces ≈ 36 card experts; on the K3 file R = 256 fits the 25 card experts' bytes and R = 512 does not without
   the margin. Proposed: R from the plan's room (the largest of 512, 256, 128 that the stage card's reserve holds),
   named in the plan's record.
3. **The read path**: the pass should run on a plan that grants the NVMe arena (floor 22.5 GB from the host share,
   §2.3), so the paged ids are read O_DIRECT once a pass and the host share stays cached; today's plan grants none.
   Stage 3's first box question is the one term: one pass of 512 on the real file, arena 0 against the floor arena,
   the NVMe bytes and wall from the tier's stats.
4. **Scoring**: one pass a line, the last 128 rows' logits through the host head as one 128-column projection — a
   change to the common `HostHead` (`crates/gpu/src/head.rs`), which projects one row a call today.

## 9. Places outside the assignment

- Multi-lane scoring (§4): several `--score` lines a pass on several state lanes; 1.75 h → ≈ 30 min `[derived]`. L.
- A router rank for K3 (`BLOOMERY_EXPERT_RANK` with a `k3trace` rank file): the host share by frequency instead of
  by id cuts the step's NVMe bytes in proportion to the routing's skew; the decode lever. M.
- `crates/gpu/src/arch/kimi_k3/placed.rs:25-26` and `mod.rs:4` say a placed body steps its prompt; they change with
  stage 2. XS.
- `crates/model/src/arch/kimi_k3/place.rs:7-9` documents `prompt_reserve` 0 and a one-token tier run; they change with
  the reserve. XS.

## 10. Stage 2 as built

The lead's answers to §8: (1) cut (b) now, the K3 walk over the common legs at `(1, R, Batch)`, 8-row mixer chunks, no
new kernel; (a), the walker's lift into `bloomery-gpu` for a second family, stays deferred until a second placed body
needs it. (2) R is a plan output. (3) The NVMe arena is granted at its floor, only on a pass plan; decode's plan is the
step's. (4) The rows projection is the common `HostHead`'s.

- **The shape** (`crates/gpu/src/arch/kimi_k3/pass.rs`): `PassShape { rows, mla_rows }`, tried widest first from
  `SHAPES` (rows 512 … 8, each with the latent mixers' inner run at 8, then 1). `generate_kimi_k3 --walk pass` plans
  each with the shape's bytes as the prompt reserve (`pass_bytes`, the same lens table the load allocates from, so a
  load whose buffers differ from the planned bytes is refused by name), the tier's batch at the shape's rows
  (`tier_batch_at`) and at least the arena's floor (`PlanInputs::with_pass_arena`), steps down on
  `PlaceError::PromptOver` alone, and prints the choice as the `prompt_pass` record (rows, mla_rows, the reserve, the
  stage card's room beside the plan's other terms, the narrowest wider shape refused and its shortfall). `[derived]`:
  the K3 file takes (256, 1), ≈ 147 MB under its 25 card experts' 162.7 MB; the K3.1 quant (512, 8).
- **The arena**: `with_pass_arena` gives the split the dial's arena, or the floor where the dial gives none (`ArenaFloor::of`: 37 slots × 610.9 MB = 22.6 GB on
  the real header) when no lever names one; at 250 GiB room the step plans paged 273.6 GB with no arena, the pass plan
  296.2 GB with 22.6 GB (`kimi_k3_meta the_prompt_pass_plan_takes_at_least_the_arenas_floor`; at 200 GiB both plans take the dial's 192.7 GB). `--walk steps`, the
  default, plans as before; the decode `--plan` lines before and after are the box's evidence (stage 3).
- **The walk**: `sched::walk(Overlap { units: 1, cols: m, port: Batch })` over `BatchLeg`: the front runs the
  AttnRes mix over m rows, the mixers in chunks, the router and the latent down per 8 columns, the card and tier places;
  the shadow the shared stack and the card leg in chunks; the back joins the tier, sums `(card + tier) + host` per
  row, norms, runs the latent up and closes the sum. `GpuModel::prompt_placed` runs a prompt as passes of the shape's
  rows; `Body::prompt` (the app) takes it when the load has the pass.
- **Scoring**: `GpuModel::score_placed(ids, n)` cuts the passes from the end so the last holds the last n rows and
  reads their logits through the host head `HOST_HEAD_ROWS` (128) rows a projection (`Head::host_logits_rows`), or a
  card head a row at a time. `generate_kimi_k3 --score FILE --walk pass` runs a line's prompt and completion but the
  last id as one call; a completion past the pass's rows is refused by name.
- **Read once** on the fixture is the tier's batches (one per tier layer per pass); the NVMe tier's one read an expert
  a pass is the real file's, and the term the stage 3 card measures (`docs/cards/k3fill-pass-read.card`).

## 11. Box run

None of this has run; each step names the line that decides it. Every command runs from this tree on the Mac. Its
remote directory is `BLOOMERY_REMOTE='~/repo/bloomery-k3fill'`, and the base tree's is `~/repo/bloomery-k3fill-base`.
The fixture tier runs on the gate lock. The real file is read only for `--plan` (headers).

**1. The gate** (`--place bp`: the A6000 is the stage, the 3090 the tier; about 20 min, bound 1200 s):

    BLOOMERY_REMOTE='~/repo/bloomery-k3fill' BLOOMERY_TIER=fixture just gate-gpu-kimik3-place

Green means every `ok` line, including the pass clauses:
- `ok pass state: 16 rows, latent 8: …` and `… latent 1: …`;
- `ok pass batches: … batches served 2L, 2 passes of L tier layers`;
- `ok pass score: …` twice;
- `ok pass refused: …` five times;
- `ok pass score, host head: … 8 of 8`;
- `ok pass cli: …` three times;
- then `PASSED: gate_kimi_k3_place …`.

The owning gates beside it come from `just affected main`, run as `tools/gate-batch.sh --tier fixture --ledger`. They
include `gate-gpu-kimik3-pp`, `gate-gpu-kimik3-e2e`, `gate-gpu-kimik3-serve` and `gate-gpu-lib`.

**2. FAIL-first.** Apply one mutant at a time, rerun step 1, check that the build log shows `Compiling bloomery-gpu`,
read the red line, and revert. The line numbers are this tree's at handoff.

| # | File:line | Change | Expected red |
|---|---|---|---|
| M1 | `crates/gpu/src/arch/kimi_k3/pass.rs:701` | `self.m * self.n,` → `(self.m - 1) * self.n,` | `FAIL pass state` on both shapes; `FAIL pass score` |
| M2 | `crates/gpu/src/arch/kimi_k3/pass.rs:1313` (in `prompt_placed`) | `.rows;` → `.rows.min(1);` | `FAIL pass batches` (served 19L, not 2L) while `ok pass state` |
| M3 | `crates/gpu/src/arch/kimi_k3/pass.rs:1373` | `let from = last.len() - n;` → `let from = last.len() - n - 1;` | `FAIL pass score` on both shapes; `FAIL pass score, host head` |
| M4 | `crates/gpu/src/head.rs:454`, a new line after `window.copy_to_host(stream, &mut data)?;` | `data.copy_within(0..hidden, hidden);` | `FAIL pass score, host head: … 7 of 8`; the card-head `ok pass score` lines stay green |
| M5 | `crates/gpu/src/arch/kimi_k3/body.rs:802` | `(p.tier_bytes() + p.pass_bytes()) as u64,` → `p.tier_bytes() as u64,` | `FAIL pass refused: a plan with no prompt reserve: None` |

**3. Decode's plan before and after** (headers only, no card load, a few seconds each). The base is `main` at this
branch's base (`a429ef3f5`), in a worktree of its own:

    git worktree add ../k3fill-base a429ef3f5
    (cd ../k3fill-base && BLOOMERY_REMOTE='~/repo/bloomery-k3fill-base' BLOOMERY_MODEL=kimi-k3 ./tools/box.sh \
      'cargo oxide build --arch $(bash tools/arch.sh) -- -p bloomery-gpu-gates --features gpu --release --bin generate_kimi_k3 \
       && target/release/generate_kimi_k3 --plan --place bp --ctx 4096') 2> base.plan
    BLOOMERY_REMOTE='~/repo/bloomery-k3fill' BLOOMERY_MODEL=kimi-k3 ./tools/box.sh \
      'cargo oxide build --arch $(bash tools/arch.sh) -- -p bloomery-gpu-gates --features gpu --release --bin generate_kimi_k3 \
       && target/release/generate_kimi_k3 --plan --place bp --ctx 4096' 2> here.plan
    diff <(grep -E '^(plan|need|place) ' base.plan) <(grep -E '^(plan|need|place) ' here.plan)

Expected: no difference. `--walk steps` is the default and plans as the step did. A difference is a finding about
`plan_with_slots` or `tier_batch_at(hp, 1)`, not a re-pin.

**4. The pass's plan** (same tree, same build):

    BLOOMERY_REMOTE='~/repo/bloomery-k3fill' BLOOMERY_MODEL=kimi-k3 ./tools/box.sh \
      'target/release/generate_kimi_k3 --plan --place bp --ctx 4096 --walk pass' 2> pass.plan

Expected `[derived]`: `prompt pass rows=256 mla_rows=1 …` with `wider=256/8` or `512/…` and its shortfall. The `plan`
line should show `arena=` the floor (about 22.6 GB) and `paged=` above step 3's. A K3.1 file at `--ctx 2048` is
expected to give `rows=512 mla_rows=8`.

**5. The one timed term**, under `docs/cards/k3fill-pass-read.card` (the lead's sitting, after steps 1–4 are green).
