# r7a predictions (V4.1 attaches the NVMe tier, attach points)

Written after the symbols were read and in the same sitting as the Mac edit (the derivation needed the code); nothing
below was settled by a box run.

## Change class per piece
| Piece | Class | Proof |
|---|---|---|
| A1, A2, decision 2 (room, arena `given`, `expert_nvme_tier(plan, room, given)`) | placement byte formula on a paged plan | `expert_nvme_tier` returns at its first line when `need <= room`, before the dial; the `given` that used to be read after that return is now read in `describe` and passed in, so on a non-paged plan nothing but the new `plan` fields moves. At `PLAN_ROOM` (263,376,371,712 B) V4.1's need (host leg 214,016,901,120 B plus the host terms, `.rec` host_experts line) is under the room: records-refresh plans move by the three new `plan` fields only (`paged=0 arena=0 arena_slots=0`). |
| A3, A4 (host run through the common `host_run`, `attach_tier`) | host dispatch on V4.1 | `nvme_tier()` returns `None` before touching anything when `nvme_expert_bytes == 0`, so `attach_tier(None)` is a no-op and the run is `HostRun::build` with the same `first`/widths/layers as before. No kernel, launch or graph node moves. |
| A5 (`Body::nvme_tier`), A6 (record fields, `nvtier` print), A7 (message text) | host / record / text | no device code. |
| ptx-scan | `generate_ds41` and `gate_e2e` identical to base (a move): no `#[kernel]` and no `cuda_module` file is touched. |

## Plan table at 48/64/96 GiB against R4b-2's floor (33 slots), derived
Slot set per slot: floor 33 slots = F + 33 s = 49,218,574,336 B, F = 26,942,887,936 B (the design's own R - arena at all
three rows), so s = (49,218,574,336 - 26,942,887,936) / 33 = 675,020,800 B.
| R | arena = R - F | S = arena / s | vs 33 | keep S - 2B (B = 16) |
|---|---|---|---|---|
| 48 GiB 51,539,607,552 | 24,596,719,616 | 36 (36.44) | holds | 4 |
| 64 GiB 68,719,476,736 | 41,776,588,800 | 61 (61.89) | holds | 29 |
| 96 GiB 103,079,215,104 | 76,136,327,168 | 112 (112.79) | holds | 80 |
The `plan` record's `arena_slots` is `arena / ArenaFloor::slot_set` and must print these S at those rooms; not run
here (the spec: `--plan` at a paging room is part b's).

## Resource timeline
Mac edit plus static tier and one box batch of owning gates at the fixture tier. No timed run, no lease, so no
resource is a bound here; the gate batch is host-CPU and card-SM bound by its own load, 10-15 min predicted from
`gate-times.tsv` medians (printed by `--dry-run` before the batch).
