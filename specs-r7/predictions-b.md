# r7b predictions (gate_ds41_nvtier: the V4.1 NVMe-tier family gate)

Written before the gate's code. Inputs: `generate_ds41 --plan --depth 512 --place a`, box, fixture tier, no lease
(plan1.log at BLOOMERY_HOST_ROOM=1G, plan2.log at F + A).
## Terms read (not predicted)
| Term | Value | Line read |
|---|---|---|
| host need (all-RAM arm's room: `need <= room` returns at `expert_nvme_tier`'s first line) | 13,005,215,744 B | plan1.log HostRoomFloor "the plan's need being" |
| F = base + 3 W | 4,881,373,184 + 3 × 1,557,135,360 = 9,552,779,264 B | plan1.log HostRoomFloor, equals its `floor` (the dial gives this plan no arena, so the refusal's floor has no arena least) |
| slot set (one slot of every paged layer) | 34,422,784 B; floor 33 slots = 1,135,951,872 B | `NvTierArenaUnder` text at a 1 B arena |
| A at S = 64 | 64 × 34,422,784 = 2,203,058,176 B (the design's number) | derived |
| paged arm's room F + A | 11,755,837,440 B | derived |
| plan at that room | paged = 6,570,147,840 B, arena = A, arena_slots = 64, card_experts 1344, residency `off` why `paged` | plan2.log:2,4 |

## Derived, judged by the gate (predicted before the first run)
- Clause 1: paged arm paged > 0, arena = A, S = 64 = nvtier.slots_per_layer = plan arena_slots; all-RAM arm paged = 0, arena 0.
- Paged ids a layer: 9 routed layers of 384 ids; 7 hold 192 on the card; the host leg is 2112 ids; the split keeps
  1 W (about one layer, 1,553,694,720 B) resident, so the arena serves 8 to 9 layers; `serves` counts the NVMe
  segment's ids and the card's victims: band 182 to 372 a layer (design §1.6), minimum above 64, so every covered layer evicts.
- Prompt call (P = 512, one batch; uniform fixture router, 512 × 6 picks over <= 384 ids touches every id: miss
  probability e^-8 an id): one fill per served id, fills ~ paged bytes / 3.9 MB ~ 1,700 [band 1,500 to 1,900]; evictions
  ~ fills − 64 × covered layers ~ 1,100 [band 900 to 1,350].
- N = 8 greedy steps: ~8 layers × 6 picks × (host-leg share ~0.5) = ~190 picks to arena ids, hit chance ~ 64 / n_l ~ 0.2:
  ~150 more misses and evictions [band 80 to 250]; slot_reads > 0; mapping_reads = 0 (R4b-2: the arena's ids read the arena).
  Totals: misses = fills ~ 1,850 [1,500 to 2,200], evictions ~ 1,250 [900 to 1,600]; the clauses need only > 0.
- Clause 3: tokens and logits FNV equal arm to arm (same bytes, same host kernel, same card split). M1 leaves it green.
- Clause 4, each refused by name before or at load: a 1 GiB room (`HostRoomFloor`, plan time); A one slot set under the
  floor, 32 × 34,422,784 = 1,101,529,088 B (`NvTierArenaUnder`, plan time); r8 on with the fixture's sidecar on the paged
  plan (`model.rs:1700`, load time, after the cards' stacks).
- Clause 5: `residency_unset(serving place)` then `at_plan(paged plan)` = `off` with why `Paged { paged: 6,570,147,840, arena: A }`.

## Resource timeline (per gate)
| Unit | Host CPU | Card SM | PCIe | Host DRAM | NVMe |
|---|---|---|---|---|---|
| one fixture load (~15 GB file) | ~10 s (22 s median incl. build in `gate-times.tsv`) | idle | ~10 GB card upload ~1 s | file in page cache | ~15 GB cold read once, ~5 s |
| prompt call P = 512, paged arm | host leg over the union | ms | small | arena 2.2 GB filled | 6.57 GB O_DIRECT reads, 1 to 3 s at 2.5 to 6.8 GB/s (QD1 to QD16) |
| 8 steps | tens of ms | ms | small | -- | ~150 × 3.9 MB = 0.6 GB, 0.1 to 0.3 s |
Wall = sum: two loads (one at a time) plus the r8 refusal's partial load, ~3 × 10 to 25 s, plus the build 10 to 60 s.
No flow lever: a correctness gate, no timed term, no lease. Predicted 1 to 3 min a batch item; the spec's
2 × (205 to 402 s) = 410 to 800 s is the upper bound (those medians include build and lock waits).
