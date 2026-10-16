# r4d1 predictions (paper, before code; base 347a86e5; host path only, no kernel: proof = ptx-scan --no-jit identical)

## 1. Safety: a victim fill never takes a slot the step reads (counting, not timing)
- The claim is unchanged: `Claim::Step`, unpinned, under the books' lock (`NvTier::claim` nvtier.rs:2064 -> `claim_with`
  nvtier.rs:370). A miss takes `pick_slot(.., arena.pins(s) || own(s))` (nvtier.rs:282, call :440): the first FREE slot,
  else the least (tick, seq) of the FILLED slots with `reading[id] == 0` (`LayerArena::pins` :660). FILLING is never taken.
  The pick turns FILLING in the same lock hold; the give-back and reads (`read_pick` :2240) run on a reader thread into that
  slot's own windows; the publish is a second hold. Only the reading thread changes.
- Step port: `experts_into` moe.rs:2401 and `experts_step_union_into_on` moe.rs:2560 read their slots after a plain `ensure`
  (no pin), so protection is by ticks: the call's ids hold the k*cols freshest ticks, and a step id is taken only after
  >= S - k*cols - pins claims land in that layer during the call (FREE first, then the oldest). S >= 33
  (`ArenaFloor.union` placement.rs:2686), k*cols = 10, pins <= LANE_THREADS = 4 (lane.rs:47), one flip in flight a layer
  = one claim (ARENA_LANE_CLAIMS placement.rs:2615). Bound holds with slack at the gate's shape.
- Prompt union: a batch's hits are pinned at the claim and its picks at the publish, each in one lock hold
  (`claim_slots_pinned` :360, `publish_pinned` :521), so a victim claim skips both; floor 2B + 1 (placement.rs:2686).
- The victim id is on the card until its flip lands: lane `serve` prepares (blocks until the fill is PUBLISHED), then waits
  the window, stages, and the boundary waits the ticket before the landing (lane.rs:608, then the ticket). So no step `ensure` meets
  a FILLING victim id (it would take a second slot and overwrite `of_id`). `prepare_victim` must stay synchronous.
- Not verified, outside this round: whether `mid-p0-s2+` (levers/src/lib.rs:245 accepts spares >= 1) puts two flips in one
  layer in flight; the slack above (S >= 33 vs 10 + 4 + 2) covers a few, a hard bound is the step port's `ensure_pinned`.

## 2. Order and deadline
- Reader `Kind`: Prompt rank 0, Victim rank 1 (reader.rs `submit`: `partition_point(|e| e.rank <= rank)`): a victim sits
  behind every queued prompt job, a later prompt job goes ahead of it.
- Decode: no prompt job queued; victim wait = thread wake + its reads = 1.12-1.42 ms (QD1, design Q5) + 0.05-0.15 ms
  hand-off [derived] = 1.2-1.6 ms. Plan: F = 21-23 victims -> 25-37 ms against the 54-86 ms issue->boundary window
  (excl-decode.card:31 gives 156-217 ms for 4 tokens) [derived], >= 1.5x.
- During a prompt chunk: the chain queues job g+1 only when job g's wait returns (moe.rs:1768), so at most one prompt job
  is queued; batches average 15.1 picks (12,302 / 815) on 16 threads, so a thread idles and takes the victim, or the first
  finished unit of a 16-pick batch does. Worst wait = one batch wall = 6.92 s / 815 = 8.5 ms + its read at the shared rate
  ~9 ms [derived from 5.57 GB/s, r4train gate_nvtier real]. A victim past its boundary is a `late` flip (the stream
  waits for the copy, `PassReport.late`): defined, bounded by that wait + stage, never a lost flip.
- No cycle: the lane thread (receive lock held) waits on the reader; reader threads wait only on the drive.

## 3. Rate
- One victim job = 1 unit = 1 pick = give-back + 3 reads in order: 1.2-1.6 ms per pick; the lane's prepare_us per
  victim does NOT fall (+4-10 %). Spreading the parts (3.134 MB / ~5.5 GB/s = 0.6 ms) needs a part-level unit that splits
  `read_pick`'s one body, and sits under the window's shadow (span 25-37 ms of 54-86 ms): 0 on tok/s -> not taken.
- Gate (paged arm, mid-p0-s1, P = 512 then 96 steps): flips ~552 [derived, w3 clean arm]; victim jobs = victims that fill
  = s x victims, s = 0.67-0.87 -> ~370-480 in the steps; victim units == jobs (one pick each); prompt jobs ~815, units
  ~12,302 [measured r4train, real]; slot_reads > 0. mapping_reads == 0 is the contract (decisions Q3) but UNANCHORED: no
  real-tier `exclusive:` line was found on the box; clause 9 tolerates mapping reads today. The run prints it first.
