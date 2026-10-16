# hostfloor — the NVMe tier arena's slot floor (host #1a) and a continuous split dial (plan S3)

Base `ee53e7c6`. Every byte figure below is the Mac planner's (`bloomery-placement`'s restated Qwen3.8 UD-Q4_K_XL
table, `crates/placement/tests/qwen38_q4_tensors.tsv`, static card figures `RTX_3090` / `A6000`, `--place a`, no
draft) unless it says otherwise. The box's figures quoted in the spec (A6000 floor 3,508,109,312 B) come from the
CLI's own context and card reading. They are not reconciled here: the formulas are what moves.

## 0. Paper first: the terms and where they come from

| term | value (Qwen3.8 Q4) | source |
|---|---|---|
| k, the routed width | 10 | `ModelTensors::experts_used` (the header's `expert_used_count`) |
| C, the widest call's columns | 1 on every served arena plan; 2 in `gate_nvtier`'s `stepunion` arm | `bloomery_levers::paged_columns` (one column on an arena plan); the step port's union is the only arena reader of several columns (`HostLayer::experts_step_union_into`, reached only from `step.rs`'s service; a prompt batch reads the mapping, `experts_union_into`) |
| L, lane claims that can land while one call reads | 1 when a stage card holds a routed expert, else 0 | `lane.rs` module doc, **The receive lock**: a victim is prepared (`prepare_victim` → `NvTier::ensure(layer, &[id])`) under one lock, one lane thread at a time, and a window-gated job then waits for the staging window under that same lock; the step port closes the window for each service's compute (`step.rs` `Closed::close` at the go, reopened at the signal). Due jobs (a landing, a flush) are all published before the pass launches (`swap.rs` **Boundaries**), and a prompt call reads no slot. So at most one lane claim lands between a service's `ensure` and its last slot read |
| one slot of a layer | Σ over its 3 routed stacks of `⌈(len + 2·4096) / 4096⌉ · 4096` | `nvtier::window_bytes`, `slot_bytes` in `of_paged` |
| Σslot, one slot in every paged layer | 151,605,248 B (48 layers); the bare expert bytes are 150,425,600 B | the window geometry above; the 1,179,648 B gap is why the floor's bytes must use the carve's geometry, not the expert bytes |
| F, the tier's room floor | base + 3 W | `expert_nvme_tier` (unchanged) |

**The floor.** One call names at most `k × C` distinct ids the arena serves (the union ensures a call's distinct ids
once, `claim_slots` dedups). A lane claim landing inside the call takes the least recently used slot, and the call's
ids hold the newest ticks after its `ensure`. So the arena serves the call without evicting one of its own ids
exactly when a paged layer has at least

    slots_min = k × C + L        (Qwen3.8, served: 10 × 1 + 1 = 11)

slots, and the least arena is `M = slots_min × Σslot` = 11 × 151,605,248 = **1,667,657,728 B**. `L` is a count
guarantee under the window rule above. It is not a pin: host #1b (a pin on the ensured ids) is the structural fix, and
with it `L` stays the slot an in-flight lane claim needs.

At plan time the split has not run yet. So Σslot is summed over every layer whose routed stacks the host leg serves
(a Host or an NVMe segment). That is an upper bound of the layers the split pages. The load's carve
(`count = arena / Σslot_paged`) therefore never falls under the plan's `slots_min`.

**One owner.** `bloomery_placement::placement::ArenaFloor::of(plan, columns)` computes `k`, `L`, Σslot (through
`arena_window_bytes`, the slot geometry's owner, with `ARENA_ALIGN` = 4096, which nvtier asserts equal to engram's
`DIRECT_ALIGN` at compile time) and `slots_min`. Three readers use it:
- the dial (`nvme_arena_of`);
- the load (`NvTier::of_paged` → `slots_of`, refusing a carve under `slots_min` by name, which closes a plan whose
  arena was set by hand or by another binary);
- a path that steps wider than the served column. `gate_nvtier`'s `stepunion` arm checks `ArenaFloor::of(plan, 2)`
  before its two-column passes. R4 passes its `slots × width` the same way.

`claim_slots`' `TooManyIds` stays the decode-time bug catcher behind it.

**Resource timeline.** This round moves no step work: the same `ensure`, fills, drops and kernels. It adds one
integer comparison at load. Per unit (one paged layer's service in a decode step), the host CPU, the card, PCIe, DRAM
and NVMe terms are each unchanged, so no flow lever applies. What moves is which plan a room gets (§2, §3), and the
only box terms that would price that are the decode and prefill numbers of the ramp band (§3.4). No box run is
needed for the floor itself: the clauses below are functional, and the lead runs them.

## 1. The defect, quantified (ee53e7c6)

On the gate plan (3090, ctx 4096, one slot), the old dial at 6 GiB gives an arena of 885,624,832 B, which is 5 slots
a layer. The first decode step's `ensure` names up to 10 ids, so `TooManyIds` poisons the tier. The band where this
hits is `F + Σslot ≤ room < F + 11·Σslot`, which is 5.71–7.22 GB on the gate plan and 3.68–5.20 GB on the A6000
plan. Under `F + Σslot`, the load already refuses ("under one slot").

## 2. The decision under the floor: refuse, naming the way to the mapping path

A room `F ≤ room < F + M` with `BLOOMERY_NVTIER_BYTES` unset is refused, by `PlacementError::HostRoomFloor`. It has
the same type, and its message keeps the prefix "the host room R B is under the NVMe expert tier's floor". Its
`floor` is now `F + M`, and it names the arena's least (slots, `k × C` ids, lanes, bytes) and
`BLOOMERY_NVTIER_BYTES=0` (the file mapping without an arena). Reasons:
1. **The one-step rule forces it.** The wide side keeps the mapping path (arena 0, §3). An arena-free low end would
   then need a second step back to 0 at the bottom.
2. **The mapping path is the worst plan at the bottom.** It would page almost the whole host leg (`H − W`) with no
   tier, so nothing drops the page cache those reads fill (Q16's class, at its largest).
3. **No regression.** That band fails today, either at the first decode token or at load.
4. **The lever stays.** A given `BLOOMERY_NVTIER_BYTES=0` is legal there (floor `F`), the user's explicit choice. A
   given arena in `(0, M)` is refused by a new `PlacementError::NvTierArenaUnder`, which names the given bytes and
   the floor. A given arena past `room − F` is still `NvTierArena`.

The callers stay valid:
- the seat's `floor_fallback` matches `HostRoomFloor` by type;
- `gate_qwen38_serve`'s band reads `HostRoomFloor.floor` off both slot counts' plans, so both floors gain the same
  `M` and the band keeps its width;
- the band's middle room is then ≥ the one-slot floor + `M`, on the dial's left branch, where
  `arena + checkpoint_bytes(1) == bare_arena` still holds;
- the `--parallel 2` refusal there keeps its prefix.

## 3. The dial

**A decision for the lead to accept.** The spec's literal shape is a dial continuous above the floor's minimum arena
with one step 0 → minimum. This design keeps one step of exactly that size but puts it at the top: a room under
`F + M` is refused (§2), and the arena falls continuously to `M` and then steps `M → 0` at `H/2 + 3W`. The
alternative, the step at the bottom, forces an arena of at least `M` on every room up to the need. The whole middle
band (the gate plan's 40 and 58 GiB, the A6000's 27 GiB) would then become arena plans, losing the draft and their
second slot until R4, and `unpaged_parallel_serves_as_asked` would go red.

### 3.1 Formula

Write `B = room − F` (the spare above the floor), `H` = the host leg's routed bytes before the split (`held`), `W` =
the heaviest layer's host bytes (`heavy`), `B* = H/2 − F`. The deep side exists when `B* ≥ M`.

| room | arena |
|---|---|
| `< F` | refused, `HostRoomFloor` (as today) |
| `[F, F + M)` on the deep side | refused, `HostRoomFloor` with the least arena (§2) |
| `[F + M, H/2)` | `B` (today's left branch, unchanged) |
| `[H/2, H/2 + 3W]` | `B* − (room − H/2) · (B* − M) / 3W`, falling from `B*` to exactly `M` (integer division, u128) |
| `> H/2 + 3W` | 0, the mapping path (today's right branch) |
| no deep side (`B* < M`) | 0 everywhere at and above `F` |

**The one step:** from `M` (at `room = H/2 + 3W`) to 0 (one byte higher). Its size is the floor's least arena,
11 slot sets.

**Continuity elsewhere:**
- On the left branch a one-byte room step moves the arena by 1 B. At `H/2` it moves by 1 B too (`B` reaches `B* − 1`,
  then the ramp starts at `B*`).
- On the ramp it moves by at most `⌈(B* − M) / 3W⌉`. Since `B* < H/2` and `W ≥ H / layers`, that is
  `< layers / 6`: 8 for 48 layers, 5 on both plans below. This is far under one slot set (151,605,248 B).
- No room has an arena in `(0, M)`.

### 3.2 Derivation

What the arena buys: decode's warm set as an LRU of the ids it serves. On a deep split that is the difference between
the nvread derivation's 26–29 and 13–19 tok/s at the 27 GiB gate room [derived, `specs/worker2/nvread/report.md` §5,
not measured here].

What it costs:
- the static host segment it takes bytes from, one for one, so the prompt reads those bytes from the drive;
- until R4, the one-column rule on that plan (no second slot, no draft, a set residency refused in the seat).

Today's rule draws the line at `room = H/2`: below it the prompt is NVMe-bound whatever the split, so the arena
takes everything above the floor; above it the static segment wins on both axes ((ii), 58 GiB: +0.7 % decode against
−70 % prefill for the max arena [derived, nvread §5]). Neither branch moves here. S3 is the cliff between them: one byte of room flips the
arena from `H/2 − F` (26.0 GB on the gate plan) to 0.

The ramp hands the arena back to the host segment over the room that the floor itself keeps for the host's own
reads, `3 W` (one layer's resident segment and the run-ahead's two-layer window). Two facts fix its place and width:
- **It sits right of `H/2`, so the left branch stays exact.** This is why every 27 GiB clause holds:
  - `gate_nvtier`'s 156-slot arena at w3;
  - `gate_qwen38_serve`'s `arena + checkpoint_bytes == bare_arena`, which needs the arena to move byte for byte with
    the floor;
  - placement's `a_27_gib_room…`.
- **Its width is in units of `W`, not of `F`.** The serving seat reserves `checkpoint_bytes(slots)` (4 GiB a slot)
  inside `F`. A ramp measured from the floor, for example "until the arena-free split pages half the host leg"
  (`room ≤ F + H/2 − W`), would end at 44.37 GB for the seat's two-slot plan. That is past the 40 GiB room where
  `unpaged_parallel_serves_as_asked` needs arena 0. `H/2 + 3W` ends at 36.93 GB there.

**What is not derived: the ramp's economics.** The width 3W is a robustness choice: a few GB of room, so the plan no
longer flips between machines that differ by a fraction of a GB. It is not a measured optimum. §3.4 names the
measurement and the alternative rules.

### 3.3 Values (Mac planner)

Gate plan (3090), ctx 4096, one slot:

| room | new arena (B) | old arena (B) |
|---|---|---|
| 5 GiB | refused, `HostRoomFloor` (floor 7,224,483,840) | refused (floor 5,556,826,112) |
| 6 GiB | refused, `HostRoomFloor` (floor 7,224,483,840) | 885,624,832 (5 slots: dies at the first token) |
| 7 GiB | 1,959,366,656 (12 slots) | 1,959,366,656 |
| 12 GiB | 7,328,075,776 | same |
| 20 GiB | 15,918,010,368 | same |
| 27 GiB | 23,434,203,136 (154 slots) | same |
| 30 GiB | 22,961,245,421 | 0 |
| 33 GiB | 7,332,069,435 | 0 |
| 34 GiB | 2,122,344,106 | 0 |
| 34.087 GiB = 36,600,934,400 (the step's last room) | 1,667,657,728 = M | 0 |
| 36,600,934,401 | 0 | 0 |
| 35, 40, 58 GiB | 0 | 0 |

The plan's other terms: `H` = 63,161,958,400, `W` = 1,673,318,400, `F` = 5,556,826,112, `H/2` = 31,580,979,200,
`B*` = 26,024,153,088, slope 4.852.

At ctx 3072 (`gate_nvtier`'s CTX), 27 GiB is the same 23,434,203,136. With two slots (ctx 4096), 27 GiB is
23,386,279,936, the ramp ends at 36,946,636,800, and 40 GiB is 0.

A6000 plan a, ctx 4096, one slot (`H` = 37,632,000,000, `W` = 998,400,000, `F` = 3,532,070,912, `H/2` = 18,816,000,000):

| room | new arena (B) | old arena (B) |
|---|---|---|
| 4 GiB | refused, `HostRoomFloor` (floor 5,199,728,640) | 762,896,384 (5 slots: dies at the first token) |
| 5 GiB | 1,836,638,208 | same |
| 12 GiB | 9,352,830,976 | same |
| 17 GiB | 14,721,540,096 | same |
| 18 GiB | 12,959,303,380 | 0 |
| 20 GiB | 3,196,776,640 | 0 |
| ≥ 21,811,200,001 | 0 | 0 |
| **27 GiB** | **0** | **0** |

The spec's A6000 example (2 slots, 27 GiB) keeps arena 0: two slots, the draft and a set residency stay as they are
there. With two slots (ctx 4096), the ramp ends at 22,153,830,400, and 27 GiB is 0.

### 3.4 What changes, and the one unmeasured term

Plans whose arena moves between 0 and > 0 against today's dial:

| plan | room band | today | new | consequence |
|---|---|---|---|---|
| any card plan, deep side | `[F + Σslot, F + M)` | arena of 1..10 slots | `HostRoomFloor` at plan time | was `TooManyIds` at the first token |
| gate (3090) | `(31.58, 36.60]` GB, about 29.4–34.1 GiB | 0, mapping path | `M`..`B*` arena | one column until R4, set residency refused in the seat, the draft refused when set; decode gains the LRU, the prompt pages up to `B*` more |
| A6000 a | `(18.82, 21.81]` GB | 0 | `M`..`B*` | same |
| every plan | `≥ H/2 + 3W + 1`, and `< H/2` | — | unchanged | — |

The rooms the gates run fall outside the ramp: 27 GiB, 40 GiB, 58 GiB, the serve band, and the A6000 at 27 GiB.

The unmeasured term is the ramp's economics: decode tok/s and pp4096 at a ramp-band room, with
`BLOOMERY_NVTIER_BYTES` at 0, `M`, the ramp's value and `B*`. A sitting of four arms at one room (`--place gate`,
`BLOOMERY_HOST_ROOM`) would place the ramp. Not run: the round is Mac-only, and no current decision waits on it. Two
alternatives were rejected now, but they are what that measurement and R4 would choose between:
- **The warm-share knee**: `arena = min(B, ρ · (H − W − B))`, the arena caching a fixed share `ω = ρ/(1+ρ)` of what
  the tier pages (ρ = 0.734 puts its peak at today's `H/2`). It is continuous with no step down to the need. But it
  puts an arena on the whole middle band: 17.7 GB at the gate plan's 40 GiB, where the prompt's paged bytes rise from
  24.1 to 41.8 GB. That breaks `unpaged_parallel_serves_as_asked`, and before R4 it costs those rooms their second
  slot and their draft (+10…+25 % decode at (ii) [derived]).
- **A slope −1 ramp** (`arena = 2B* − B`, down to `M` at `H − F − M` ≈ 55.9 GB): the most continuous, with the same
  middle-band consequence. 40 GiB would carry 14.6 GB.

Once R4 serves `slots × width` columns on an arena plan, the middle band's arena costs no feature. Then the ramp can
widen toward the knee after that sitting.

## 4. The gates' plans

| gate, room | arena before → after | stays valid |
|---|---|---|
| `gate_nvtier` paged arm, 27 GiB (ctx 3072, one slot) | 23,434,203,136 → same; 154 slots on the Mac's table, 156 at w3 on the box's | yes (left branch) |
| `gate_nvtier` stepunion, 27 GiB, two slots | 23,386,279,936-class → same | yes; its new pre-check `ArenaFloor::of(plan, 2)` = 21 slots ≤ its carve |
| `gate_nvtier` `refuse floor`, 1 GiB | `HostRoomFloor` → same type | yes |
| `gate_nvtier` `refuse slot`, the 1 B arena | "under one slot" → "carves 0 slots … under the floor of 11 slots" | the clause's expected text moves, in this round's diff |
| `gate_qwen38_serve` 27 GiB (one slot after the rule; floor + 4 GiB checkpoints) | left branch | `arena + checkpoint_bytes == bare` holds |
| `gate_qwen38_serve` 40 GiB `--parallel 2` (floor + 8 GiB) | 0 → 0 (ramp ends at 36.93 GB + 0) | yes |
| `gate_qwen38_serve` band (between the slot floors) | left branch | the floors both gain `M`; the fallback identity and the `--parallel 2` refusal hold |
| placement `the_one_column_rule_reads_the_arena` (27 / 40 GiB) | unchanged | yes |
| placement `a_58_gib…`, `a_paged_plans_load_check…`, `…pool_fits…`, `…cache_budget…`, `…run_ahead_window`, `…churn_pool` | unchanged | yes |
| placement `a_room_under_the_floor_is_refused_by_name` | floor `F` → `F + M` | re-pinned, `PIN(2026-10-09)` |
| placement `the_split_dial_picks…` | the `H/2` jump → the ramp and the one step | re-pinned, `PIN(2026-10-09)` |

## 4b. The room past the need at the serve gate's rooms (roomneed's prompt-cache default)

roomneed (4144b1d8, not in this tree) sets the prompt cache's default to 0 on any plan the NVMe tier splits, because
the split fills the room past the floor. That stays true under the new dial, at every room it plans:
- the split keeps `room − arena − base − 2W` of host experts, in whole experts a layer;
- so `room − HostNeed::bytes()` (arena included) is the split's rounding residue, whatever the arena;
- the dial only moves bytes between the arena and the host segment, never into the room past the need. The ramp band
  splits today too (arena 0), so its residue is the same kind.

Mac planner, gate plan (3090), ctx 4096, the seat's machine emulated with a 4 GiB-a-slot checkpoint reserve on the
host (the seat's `CHECKPOINTS_RESERVE`):

| room | slots | arena (B) | the need with the arena (B) | room past the need (B) |
|---|---|---|---|---|
| 27 GiB | 1 (the flagless server after the one-column rule) | 19,139,235,840 | 28,990,824,448 | 204,800 |
| 27 GiB | 2 | 14,796,345,344 | 28,990,210,048 | 819,200 |
| 40 GiB | 2 (`--parallel 2`) | 0 | 42,946,658,304 | 3,014,656 |
| the band's middle, 13,690,896,384 (new floors: one slot 11,519,451,136, two slots 15,862,341,632) | 1 | 3,839,102,976 | 13,690,691,584 | 204,800 |
| the same room | 2 (`--parallel 2`) | refused, `HostRoomFloor` (floor 15,862,341,632) | — | — |

The band itself moves up by `M` (old floors 9,851,793,408 and 14,194,683,904, middle 12,023,238,656). Both slot
counts' floors gain the same `M`, so its width is unchanged.

## 5. FAIL-first

Mac, `bloomery-placement`. Each test is red on ee53e7c6's lines, shown by editing the fixed line back:
- `a_room_that_leaves_under_the_floors_slots_is_refused_by_name`: the gate plan at `F + 9·Σslot` (k − 1 slots) and
  at `F + 10·Σslot` (slots_min − 1) is refused by `HostRoomFloor`, naming 11 slots, 10 × 1 ids, 1 lane claim and the
  bytes; at `F + M` it plans the arena `M`. Old: both plan arenas of 9 and 10 slots.
- `a_given_arena_under_the_floor_is_refused_by_name`: `NvTierArenaUnder` at `M − 1` names the floor; `M` and 0 are
  taken; a given 0 under `F + M` plans the mapping path. Old: `M − 1` is taken.
- `the_dials_one_step_is_from_the_least_arena_to_none`: the gate plan at `H/2 + 3W` plans `M` and one byte higher
  plans 0, through the planner. The step's size is 11 slot sets. Old: both 0, with the jump at `H/2`.
- `the_dial_moves_by_its_slope_bound_elsewhere`: on both plans' terms, at every breakpoint ±1 and on a 64 MiB grid
  from `F` to the need, a one-byte room step moves the arena by at most `⌈(B* − M)/3W⌉`, except at the step. No room
  has an arena in `(0, M)`, and the A6000 at 27 GiB is 0. Old: the `H/2` jump (26.0 GB) is red.

Box (the lead runs them; the mutants are in `specs-hostfloor/`):
- `just gate-gpu-lib`: nvtier's `a_carve_under_the_floor_is_refused_by_name`. A budget of `slots_min − 1` slots is
  refused naming the slots, `k × C` ids and the lane claims, and `slots_min` carves `slots_min`.
- `just gate-gpu-nvtier`: `refuse slot` reads the new text ("carves 0 slots"). The `stepunion` arm refuses by name
  an arena whose carve is under `ArenaFloor::of(plan, 2)`. The `refuse` clause adds:
  - `refuse floor-k`: the arena `(k − 1)·Σslot`, refused at load by name;
  - `refuse floor-lane`: the arena `(slots_min − 1)·Σslot`, refused;
  - `accept floor`: the arena `slots_min·Σslot` loads with `slots_per_layer == slots_min`, and its carve's slot set
    equals the placement's (the geometry's one owner).

The box mutants (`git apply specs-hostfloor/mN.patch` on this tree, one at a time; each predicts its reds, every
other clause green):

| mutant | the change | `just gate-gpu-lib` | `just gate-gpu-nvtier` |
|---|---|---|---|
| `m1.patch` | `slots_of` refuses only a carve of 0 (the old check) | red: `a_carve_under_the_floor_is_refused_by_name` (11·Σ − 1 carves `Ok(10)`) | red: `refuse floor-k`, `refuse floor-lane` (both arenas load); `accept floor` green |
| `m2.patch` | `of_paged`'s floor without the lane claim | green (`slots_of` itself unchanged) | red: `refuse floor-lane` (10 slots load); red: `refuse floor-k` (its text names a 10-slot floor with 0 lane claims) |
| `m3.patch` | nvtier's window one page wider than the placement's | green (the window tests check alignment and span, not size) | red: `accept floor` (the floor's bytes carve 10 slots and a larger slot set); red: `refuse floor-k`, `refuse floor-lane` (they carve 8 and 9, not 9 and 10) |
| `m4.patch` | `NvTierStats::slots_per_layer` reads 0 | green | red: `accept floor`; the `stepunion` arm ends the run by name ("carves 0 slots a paged layer, under its two-column floor") |

`check-comments` passes `specs-hostfloor/` (no retired word). The patches are diffs against this tree, and they
apply with `git apply --check`.
