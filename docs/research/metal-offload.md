# Models larger than a Mac's GPU working set: what offload means on unified memory (facts, carry-over, paper design)

Round `mac-offload`, 2026-10-25, main 85f17e393. Research only: no repo code changed, nothing run on the box, no sysctl set, no
download. Tags: `[measured]` a printed number with its command in this file; `[derived]` arithmetic on named inputs;
`[unverified]` not seen here. Probes are scratch programs
(Swift and Python, not in the repo); every number below names the probe that printed it. Re-reviewed 2026-10-25: the last
section lists each finding and where it landed.

Machines: lead Mac = Apple M1 Pro 32 GB (macOS 27.0, shared with this session's other work); second M1 Pro = Apple M1 Pro 32 GB,
macOS 27.0.1 (a Qwen3-30B download wrote ~10 MB/s through every disk number below, and ~6 GB of its swap was
already in use); M4 = Apple M4 Pro 48 GB, macOS 27.0.1 (its swap was 24-25 GB used of
26 GB, pressure level 1, 15 GB of disk free).

## 0. Summary

1. The GPU working-set limit is **0.78 x RAM** on both measured sizes (M1 Pro 32 GB: 24.96 GiB = 26.80 GB; M4 Pro 48 GB:
   37.44 GiB = 40.20 GB), `iogpu.wired_limit_mb` is 0 (system default) on all three, `hasUnifiedMemory` is true, and one buffer
   may be at most 0.585 x RAM (18.72 GiB; 28.08 GiB); the fraction is [unverified] below 32 GB and above 48 GB. Apple documents the
   limit as a performance guideline, not a cap.
2. Past the limit was **not observed**: on the lead Mac the memory-pressure rule stopped the allocation probe at 15 GiB, 10 GiB
   short of the limit, because the GPU's wired buffers push other processes' pages into the compressor first.
3. A cold SSD read on both Macs plateaus at **6.7-6.8 GB/s** at queue depth 4 or more and 1 MiB or more a read; a single 3 MiB read
   in flight gets 4.7-4.8 GB/s; reads through `mmap` page faults get 0.25-0.9 GB/s. Reads land in a shared Metal buffer at the same
   rate while the GPU streams, and the GPU loses at most 6 % ([measured], section 1.4).
4. A GPU kernel that spins on a flag the host stores later **does not reliably see it** on this macOS ([measured], 1.5; on the
   M4 it saw none in 300 ms): the CUDA handoff page cannot be ported. What works is the host polling a word a *finished* dispatch
   wrote, seen before its command buffer ends (1.6-13 us, probe C), and committing the next pre-encoded command buffer (1.2-13 us
   to its first instruction).
5. Recommended design: option (b), the CUDA NVMe tier's shape with the memory kind `Pool` that `tierengine.md` section 3(h) already
   designed for GB10 (the arena is the card's memory), on one common device-free tier core; each layer's command buffer is cut at
   its router and the host polls the routed ids. Predicted Qwen3.8-Flash-Next UD-Q4_K_XL decode, batch 1, SSD wait in the chain
   (sum), eta 0.90, as a band under a working-set ceiling no probe has reached (section 3.3): ~~11.8-13.4 tok/s on the 32 GB M1
   Pro (quiet machine), 21.0-23.3 on the 48 GB M4 Pro~~ **8.2-9.2 on the M1 Pro at the lead Mac's 15 GiB stop (arena 9.1 GB, S 60)
   up to 11.8-13.3 with the limit reached (S 131); up to 20.6-22.8 on the M4 Pro (S 220; its low end is [unverified])**. At eta
   0.75, the Qwen3.8 Metal kernels being unwritten: up to 10.7-11.9 and 18.4-20.2. ~~17.6-18.6 and 28.5-29.7 with 70 % of the SSD
   time hidden under GPU work~~ is withdrawn: the first flow lever is the hits' compute under the miss's own read inside a layer
   (12.8-14.4 and 21.9-24.2); the lookahead across layers hides at most 34-43 % of the SSD time (14.4/16.1 and 23.7/25.9 at recall
   1). The first offload customer is Qwen3.6 on a 16/24 GB Mac, after the design doc's Phase 1 (section 3.5).
6. The CPU leg (c) is bounded by +8 % (4 threads) to +24 % (8 threads) more bus bandwidth than the GPU alone and needs NEON kernels
   for five quant types; (d) the OS pager is 3.3x slower per expert than a read into a slot. Neither is first.

## 1. Part 1: measured

### 1.1 Limits (probe `limits.swift`, a `MTLDevice` query plus `sysctlbyname`)

    lead Mac  (M1 Pro 32 GB)  recommendedMaxWorkingSetSize=26800603136 (24.96 GiB)  ratio 0.7800002  maxBufferLength=20100448256 (18.72 GiB)  hasUnifiedMemory=true  iogpu.wired_limit_mb=0
    second    (M1 Pro 32 GB)  recommendedMaxWorkingSetSize=26800603136 (24.96 GiB)  ratio 0.7800002  maxBufferLength=20100448256 (18.72 GiB)  hasUnifiedMemory=true  iogpu.wired_limit_mb=0
    M4 Pro 48 GB              recommendedMaxWorkingSetSize=40200896512 (37.44 GiB)  ratio 0.7800001  maxBufferLength=30150672384 (28.08 GiB)  hasUnifiedMemory=true  iogpu.wired_limit_mb=0

`maxThreadgroupMemoryLength` 32768 on all three. Larger Macs: the fraction is [unverified] above 48 GB (a community post says
"I believe Apple has a hard coded limit of 75% of physical memory", https://developer.apple.com/forums/thread/732035; other guides
cite 75 % or two thirds); this file uses 0.78 and says so. Apple's line
(https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/recommendedmaxworkingsetsize.md): "An approximation of
how much memory, in bytes, this GPU device can allocate without affecting its runtime performance." The code already reads it as
the integrated device's pool: `crates/metal/src/census.rs:26-45` (`total_bytes` = the limit, `free_bytes` = limit minus
`currentAllocatedSize`, `integrated: true`).

### 1.2 What happens past the limit (probe `pastlimit.swift`, lead Mac only)

One 1 GiB shared buffer a step, filled with an incompressible pattern (a quantized weight file does not compress), then one
command buffer reading all buffers so far. Stop rules: allocation refused, command-buffer error, swap used up by 1 GiB, pressure
level 2, `currentAllocatedSize` past limit + 4 GiB, or 26 GiB. Start state: swap 2.46 GiB used, free 5.05 GiB, compressor
3.45 GiB, pressure 1 (the machine runs this session and other rounds).

    step | currentAllocatedSize_GiB | read_gpu_s | GB/s(gpu) | status    | swap_used | free_GiB | compressor_GiB | wired_GiB | swapouts | pressure
    1    | 1.02                     | 0.014      | 77.3      | completed | 2.46      | 3.98     | 3.45           | 4.36      | 0        | 1
    5    | 5.02                     | 0.039      | 136.5     | completed | 2.46      | 0.13     | 3.45           | 8.38      | 0        | 1
    10   | 10.02                    | 0.070      | 153.5     | completed | 2.46      | 0.08     | 4.29           | 13.23     | 0        | 1
    15   | 15.02                    | 0.102      | 157.7     | completed | 2.46      | 0.10     | 6.72           | 18.10     | 0        | 2
    stop: memory pressure level >= 2 at 15 GiB (stop rule); residency set over all 15 buffers: allocatedSize=15.00 GiB, request+read 0.109 s, completed

What it shows [measured, `logs/pastlimit-lead.txt`]: (a) allocation never refused and no command buffer failed up to 15 GiB;
(b) the GPU wires a buffer when the first command buffer binds it (wired grows by 1 GiB a step and stays up after the buffer
completes); (c) the system makes room by compressing other processes' pages (compressor 3.45 -> 6.72 GiB, free pinned near 0.1 GiB
from step 5) with no swap-out until pressure level 2; (d) a read kernel over 15 GiB runs at 150-158 GB/s, so residency costs
nothing once wired. **Past 24.96 GiB: [unverified]**, needs an idle Mac; the usable arena on a machine in daily use is set by the
other processes (15 GiB on this one), not by the limit.

### 1.3 SSD reads, cold (probe `ioprobe.swift`)

`pread` with `F_NOCACHE` into 16 KiB-aligned buffers, 16 KiB-aligned random offsets, threads = queue depth, a fresh offset every
read. Cold is checked, not assumed: the probe maps the file read-only, calls `mincore` over all of it and draws only spans with no
resident page (it reports the resident share, and a case it cannot draw cold is printed `invalid`). The first second M1 Pro's run read a
just-downloaded file and printed 25-41 GB/s, which is the page cache; it is discarded (`logs/io-d1.txt`).

    cold GB/s                      M4 Pro 48 GB (Qwen3.6-35B-A3B-Q4_K_M, 21.2 GB)   M1 Pro 32 GB, second (Qwen3-30B-A3B-Q4_K_M, 18.6 GB)
    seq 16 MiB, QD 1               5.74 (run 1)                                     5.83
    seq 16 MiB, QD 4               6.46 (run 1), 6.58                               6.38-6.72
    random 1 MiB,  QD 1 / 4 / 16   3.05 / 6.40 / 6.74                               2.85 / 6.43 / 6.28
    random 3 MiB,  QD 1 / 4 / 16   4.73 / 6.73 / 6.74                               4.78 / 6.81 / 6.79
    random 6 MiB,  QD 1 / 4 / 16   5.45 / 6.77 / 6.54                               5.35 / 6.77 / 6.84
    mmap, touch 16 KiB pages of random 3 MiB chunks, QD 1 / 4   0.25 / 0.63        0.42 / 0.93
    latency p50, 3 MiB QD 1        0.66 ms                                          0.66 ms

Sources: `logs/io-m4.txt` (run 1, 11:16: the probe had no residency check yet, but the file was ~0 % resident, as the 8.6 % that
the run's own two 1 GiB mmap cases left behind shows; 1024 MiB a case), `logs/io-m4-cold.txt` (11:33, mincore-checked, 8.6 %
resident, cold spans only, 2048 MiB a case), `logs/io-d1-cold.txt` (file 80 % resident, cold spans only; the download wrote
~10 MB/s throughout). `logs/io-m4-b4096.txt` (11:17, 7.1-7.5 GB/s) is **discarded**: it ran after run 1's mmap cases had warmed
~9 % of the file and drew offsets without the check, and 9 % of the bytes at memory speed lifts 6.7 to ~7.4. Sequential QD 1 on
the M4 could not draw a cold 2 GiB run in the 11:33 run (printed `invalid`); 5.74 is run 1. The M1 Pro's SSD is as fast as the M4
Pro's here: both Macs have one ~6.7 GB/s plateau. The CUDA box's drive plateau for
comparison is 6.77 GB/s at QD 16 and 2.51 at QD 1 on 1.045 MB parts (`tierengine.md:362-364`).

### 1.4 A fill lands in a Metal buffer at the drive's rate while the GPU streams (probe `overlap.swift`, M4)

    round 1 | GPU alone      |   248.2 GB/s | 664 x 1 GiB reads
    round 1 | SSD alone QD4  |    6.98 GB/s | p50 1.82 ms | 6657 preads of 3 MiB into the shared buffer
    round 1 | GPU + SSD      | GPU 234.4 GB/s (94 % of alone) | SSD 6.86 GB/s (98 % of alone)
    round 2 | GPU + SSD      | GPU 236.3 GB/s (97 % of alone) | SSD 7.01 GB/s (100 % of alone)

This probe draws offsets without the residency check, and the file was ~9-27 % resident from the earlier mmap cases: the 6.9-7.0 GB/s
SSD figure is up to ~5-10 % above the cold plateau; the concurrent-to-alone ratios are within one run and unaffected. The M4 Pro's
GPU read rate is 243-248 GB/s [measured]; the M1 Pro's 147-160 GB/s on the lead Mac
(`logs/cpugpu-lead.txt`: 160.4 and 147.2; `logs/pastlimit-lead.txt` reads 119-159 over its 15 steps; the lead Mac was busy). Prediction written before the M4 run: GPU 190-240
(missed high), SSD 6.5-7.5 (hit), concurrent GPU 93-100 % and SSD 90-100 % (hit). **Reading the file into a slot of a shared
`MTLBuffer` is the same call as reading it anywhere; nothing is copied; the GPU is not slowed more than 6 %.**

Correctness of the in-place fill (probe `coherence.swift`, M4): 500 rounds of `pread` (3 MiB, `F_NOCACHE`) into a shared-buffer
slot that held a stale pattern, then a command buffer that XORs the slot against the host's XOR of the same bytes:
`mismatches 0, command buffer errors 0` (a pread 0.62 ms mean, the kernel 0.19 ms). [measured]

### 1.5 The host/GPU handoff a paging layer needs (probes `latency.swift`, `spin.swift`)

A paged layer must get its routed ids to the host, fill the misses, and let the layer's expert kernels run.

    lead Mac M1 Pro                                                                    M4 Pro
    A commit + waitUntilCompleted (tiny kernel)         p50 175-187 us  p95 240-315    p50 89 us  p95 123
    B commit -> a word the kernel wrote is visible to a polling host   p50 3.1-3.2 us p95 9.3     p50 1.6 us  p95 3.5
    C a dispatch names ids, the host sees them, releases the next kernel queued in the same buffer, it runs   p50 3.2 us  p95 9.0   p50 1.5 us  p95 2.1   (timeouts 0 on both)
    D the same B after an idle gap of 0.1 / 1 / 5 / 20 / 100 ms        2.8 / 3.1 / 7.2 / 11.0 / 13.2 us p50    1.2 / 2.0 / 2.3 / 2.5 / 6.2 us p50

B is the lowest the host can observe an id; it holds after 100 ms idle (the GPU wakes in microseconds). C is B's loop plus the
host's release of a second kernel queued behind the first in one command buffer (`logs/latency-lead.txt`, `logs/latency-m4.txt`,
2,000 trials each). Then the failing case: a kernel already running spins on a word the host stores later (`spin.swift`, a counter
as the value so a stale equal cannot pass, a host deadline of 300 ms, the kernel gives up after 20 M loads):

    v1 relaxed atomic load loop, host stores at +0 us    flag store -> kernel done: p50 116939 us   kernel gave up: 8 of 20
    v1 ... host stores at +2000 us                       p50 56325 us   6 of 20
    v2 volatile plain load, +0 / +2000 us                p50 98202 / 61011 us   6 of 20 / 0 of 20
    v3 atomic_fetch_or(0) (a read-modify-write), +0 / +2000 us   p50 40197 / 21496 us   0 of 20
    V5 MTLSharedEvent: a queued command buffer waits for the host's signal   signal -> kernel done p50 100.8 us (p95 162), 350 us after 2 ms idle

On the M4 the same probe found that the kernel's own store to `started` was not visible to the host within 300 ms (`kernel
never started`); with the host waiting 3 ms instead (`spin nostart`) no variant, the read-modify-write included, finished within
the host's 300 ms in 20 of 20 trials (a kernel that had seen the store would have ended at once), while V5 on the M4 took p50
61-74 us (p95 70-83), 445 us after a 2 ms wait (`logs/spin-m4.txt`). An earlier variant of the probe that reset the flag
instead of counting printed 0 us (probably the previous trial's stale value read by the GPU), and a first `coherence` run measured
~98 ms from the host's store to completion: both are discarded. **A running kernel sees a later host store only after tens to
hundreds of ms, or never within the limit.** A dispatch's store, on the other hand, is seen by the host before its command buffer
ends: in C the second kernel cannot finish before the host stores its release, and none gave up in 2,000 trials on either Mac
(timeouts 0) [measured]. C works because the release lands before the waiting kernel starts; a fill takes ~0.5 ms, so the
kernel would already be running and the spin result applies (host store at +2 ms, seen after 56 ms, v1 above) [derived]. The
design therefore ends the first command buffer at the router and releases the next by a new commit. Measured on the lead Mac (M1
Pro) and the M4 Pro, macOS 27.0 / 27.0.1. The CUDA design's handoff (the page of `crates/gpu/src/host/page.rs:1-40` and the
`op_wait_geq` ops of `step.rs:17`, `lane.rs:227`) assumes the opposite. The working primitives:
GPU -> host: the host polls a word the kernel wrote (B: 1.6-3.2 us p50); host -> GPU: **commit the next command buffer, encoded
ahead** (B: 2-13 us to its first instruction), or a shared event (~100 us). `waitUntilCompleted` per layer would cost 48 x
90-190 us = 4-9 ms a token. Polling plus commit puts B + C = 3.1 us (M4 Pro) and 6.3 us (M1 Pro) on the chain a cut [derived: sums
of p50s; C's loop already holds its own commit-to-seen leg, so the sum may count that leg twice]; with the tier's books (~15 us a
layer, paid only when a tier claims slots) a layer's sync is ~21 us and 48 layers ~1 ms [derived].

### 1.6 CPU and GPU together, and the OS pager as a buffer

CPU threads and the GPU streaming separate 1 GiB buffers (probe `overlap.swift cpugpu`, lead Mac, busy):

    CPU 4 threads | GPU alone 160.4 | CPU alone  88.2 | together GPU 126.9 + CPU 45.8 = 172.6 GB/s
    CPU 8 threads | GPU alone 147.2 | CPU alone 108.0 | together GPU 123.8 + CPU 75.0 = 198.8 GB/s

The shared DRAM bus gives 172.6/160.4 = +8 % (4 threads) and 198.8/147.2 = +35 % (8 threads) over the GPU alone in these two noisy
runs on a busy machine; the runs' own GPU-alone figures differ (160.4 and 147.2 GB/s; the M1 Pro's spec is 200 GB/s), so +35 %
mixes baselines and the 8-thread run is +24 % against the 160.4 [derived]. That is the ceiling of option (c) on this chip, before
any kernel runs below the stream rate.

The OS pager under the GPU (probe `overlap.swift mmapbuf` and `mmapexp`, M4; cold random file ranges):

    pread 1 GiB into a shared Metal buffer (16 MiB reads, QD 1): 0.19 s = 5.64 GB/s; then the GPU read 0.021 s
    GPU read over a file-mmap bytesNoCopy buffer, cold:  wall 0.537 s = 2.00 GB/s end to end; gpu time 0.005 s; second read 0.005 s
    one layer's 10 experts (10 x 3 MiB cold random):
      pread arm (4 threads into slots, then the GPU reads them):                  p50 4.95 ms, p95 5.55 ms
      mmap arm (10 bytesNoCopy buffers over the file mapping, one command buffer): p50 16.36 ms, p95 23.53 ms; 10 makeBuffer calls 0.051 ms

`makeBuffer(bytesNoCopy:)` over a file mapping works, and the driver faults and wires the whole range before the command buffer
starts (the GPU itself never faults): option (d) runs, at 3.3x the per-layer cost and with no control over what stays. Predictions
written before: pread 4.8 ms (band 4-7, hit), mmap 16 ms (band 12-40, hit), mmapbuf end-to-end 0.25-0.6 GB/s (missed: 2.0).

## 2. Part 2: the CUDA host code, what carries over

Reading `tierengine.md` first (648 lines, by range): section 3(h) (`:264-327`) is the design for a CPU+GPU single pool (GB10), the
Mac's shape; section 4 (`:471-502`) the cost model; 3(j) (`:360-386`) the read units. Piece by piece:

| piece | path:line | device-free? | on a Mac | what unified memory changes |
|---|---|---|---|---|
| tier books: `LayerArena`, `SlotBook`, `pick_slot` (LRU, transient first), `claim_fill`, `publish`, `Books`, heat counts | `crates/gpu/src/host/nvtier.rs:412,562,702,819,1177,1743,1815` | yes in substance; the file imports `cuda_core::CudaContext` (`:106`), `GpuError` (132 uses) and `model::moe::{Claim, TierSlots, ...}` (`:109`) | carries over unchanged as policy | nothing: a slot is a window of the arena either way |
| arena memory: `ArenaMem` (`:139`), `ArenaKind::{Anon, Pinned}` (`:307`), `ArenaMem::pinned` (`:189`, `cuMemHostAlloc`) | `nvtier.rs:139-200,307-321,2126-2130` | the only CUDA-bound line group of the tier core (plus `issue_card_hold`) | a third kind `Shared`: an `AnonMap` wrapped by `makeBuffer(bytesNoCopy:)`, or the `MTLBuffer`'s own pages | no `MADV_DONTNEED` give-back (the refill overwrites in place; `tierengine.md:301`); one buffer is at most 0.585 x RAM, so one buffer per layer group |
| fill reader: `NVTIER_FILL_THREADS` 16, rank queue, one `pread` a part | `nvtier.rs:127`, `nvtier/reader.rs:1-16,321` (792 lines) | yes (`GpuError` only) | carries over; thread count is the measured queue depth (QD 4 reaches the plateau on both Macs) | the unit stays a part (3 a fill, ~1 MB each: QD 3 of 1 MiB parts = 5.2-5.7 GB/s [derived between QD 1 and QD 4 above]) |
| the O_DIRECT reader: `DirectFile`, `AnonMap`, `aligned_span`, `DIRECT_ALIGN` | `crates/engram/src/direct.rs:29,45,143,156` | yes | already runs: `crates/os/src/lib.rs:286` opens with `F_NOCACHE` on macOS | `DIRECT_ALIGN` is 4096 and Apple pages are 16 KiB; the probe above used 16 KiB alignment, 4 KiB-skewed `F_NOCACHE` reads are [unverified] |
| the seam to the model: `TierSlots`, `Claim`, `Ticket`, `TierError` | `crates/model/src/moe.rs:2479,2290`; impl `nvtier.rs:3641` | device-free but in `crates/model`, which the Mac cannot build (`workstation.rs:74` reads `/sys/fs/cgroup`; `tools/recipes.py pure-crates` rejects it) | the seam types move with the tier core | none |
| the card leg: `ArenaLeg`, `issue_card_hold`, `CardTicket`, `CardHold::offsets`, the `_at` kernel entries | `crates/gpu/src/host/arena_leg.rs:1-31`, `nvtier.rs:1205,1258,2813` | CUDA-bound (streams, mapped tables, `kquant::sel`) | the template for the Metal leg: ids -> slot byte offsets -> expert kernels | on GB10 and here the card reads the arena in place: a hit costs the expert's bytes at the GPU rate, no copy |
| handoff page and wait ops | `crates/gpu/src/host/page.rs:1-40`, `tier_cpu.rs:1-13` | page layout is arithmetic (no cuda); the protocol is a GPU spin-wait | **does not port** (section 1.5) | replaced by poll + commit |
| card mover: `SwapMachine`, lanes, flips; `FileSwap`; rule | `swap.rs:1-30`, `swap_source.rs:1-50`; `crates/runtime/src/swaprule.rs:522` | the machine is CUDA-bound (streams, events); the rule is device-free | the mover collapses: with one pool there is no second level to flip between (`tierengine.md:283-284`: the card mover is `off`, why `Unified`); the rule's heat could rank arena eviction | `Hosting::Unified` already exists (`placement/workstation.rs:1164`) |
| expert tiers on further cards, `TierNode`, links | `tier.rs:1-64` (2912 lines) | CUDA-bound | not applicable to one device | - |
| census | `crates/gpu/src/host/census.rs:1-17` | yes | carries over as the counters of record | - |
| host set residency (populate, `mlock`, `mincore`) | `gpu/src/host/residency.rs:17,32`, `model/src/placement/host_lock.rs:305,648,982` | `crates/model`, Linux paths; `MADV_POPULATE_READ` absent on macOS (`host_lock.rs:982-991` names it) | not needed for experts (they are read by `pread`, not mapped); the PLE / row tables use `engram` | `MTLResidencySet` replaces `mlock`: `crates/metal/src/residency.rs:1-24` |
| row tables from NVMe: `RowTable`, `Prefetcher`, `Site` | `crates/engram/src/lib.rs:812`, `prefetch.rs:124,254,324`, `rows.rs:1-22` | yes | PLE (28.8 GB, 16 rows of 90 B a token) and V4.1's engram (84.6 GB, 110 B rows) | the reader's `MADV_POPULATE_READ` is Linux only; page faults on 16 KiB pages cost ~65 us each (1.3: mmap 0.25 GB/s), so the helper-thread prefetch is what keeps it off the step |
| placement: `expert_nvme_tier`, `nvme_arena_of`, `ArenaFloor`, `HostNeed` | `crates/placement/src/placement.rs:3861,3813,3591`; `workstation.rs:854` | yes, a pure crate; integrated devices already plan against one pool (`devices.rs:34-40`, `workstation.rs:48-51,869`) | carries over; `ARENA_ALIGN` is 4096 (`placement.rs:3544`) | the Metal census reports `limit - currentAllocatedSize` as free (`census.rs:40`), which counts the other processes as zero and breaks the integrated contract, "the host's room" (`devices.rs:33-36`); on a busy Mac it overstates the room by the others' footprint (1.2: 10 GiB on the lead Mac) |
| qdot (CPU expert dots) | `crates/qdot/src/lib.rs:1-60`, `generic.rs:32,296` | host code, but AVX2 (`target_arch = "x86_64"`) plus a scalar `dot_scalar` | no NEON kernel exists (no `neon`/`aarch64` in `crates/qdot/src`) | the CPU leg here shares the GPU's DRAM (1.6) |
| the gate: paged arm vs all-RAM arm, logits bit for bit | `crates/gpu-gates/src/bin/gate_nvtier.rs:1-40`, `bin/shared/nvtier_family.rs:1-33` | GPU gate | pattern to copy | real-file only: `justfile:2758` runs `real-only.sh`, the all-RAM arm needs 105.8 GB of host (`tools/gate-batch-resources.tsv:44`) |

Two observations that decide the design. The tier core is one coupling away from device-free: the memory kind and the card hold
(`CudaContext` appears at `nvtier.rs:106,142,189,309,316` and the pinned arena at `:151`), so lifting it into a crate the Mac
builds is a split along an existing seam, not a rewrite. And the Metal kernels already read an expert through one index:
`crates/metal/kernels/experts.metal:25` ("every other slot reads stack + ids[slot] * expert bytes") and `:177-182`
(`index = (expert * rows + row) * per_slot + local`), so a slot table replaces `ids[slot]` at one place per kernel.
Weights are loaded today one tensor to one buffer by copy (`crates/metal/src/arch/qwen3moe/arena.rs:242,341-348`).

## 3. Part 3: paper decomposition and design

### 3.1 Terms from the headers (Qwen3.8-Flash-Next UD-Q4_K_XL, 111.3 GB; `py/q38bytes.py` over the header dumps of the four shards)

    attention 2.239 GB | shared expert 0.251 | router 0.252 | hyper-connection 0.688 | indexer 0.039 | ssm/delta 0.643 | head+output hc 0.682
    routed experts 77.018 GB (24,576 experts, mean 3.134 MB; gate/up Q4_K 0.9216 MB each, down Q5_1 1.2288 MB, Q8_0 1.7408 in 5 layers)
    PLE table (per_layer_token_embd, IQ4_NL, 90 B a row) 28.80 GB + projections 0.035 | token_embd 0.675 GB (1 row a token)
    sum of listed tensors 111.324 GB (the README's 111.3 GB)

A token reads the dense set D = 4.829 GB and 48 x 10 experts = 1.504 GB: **6.333 GB** [derived]. `py/active_bytes.py` run on the
Qwen3.6 UD-Q4_K_XL header gives 2.753 GB a token against the 2,884 MB in `plan-triage.md`; Qwen3.6 Q4_K_M decodes at 85.6 tok/s
on this M4 Pro (`mac-m4pro-xeng-2026-10-25.md`), so the Metal decode kernels run at 223-235 GB/s of the 245 GB/s stream rate
(91-96 %) [derived: file ratio 0.945 x 2.753 = 2.6-2.75 GB]. That is Qwen3.6's figure. Qwen3.8 has no Metal kernels yet (15th
in the design doc's cover order), and its hyper-connection, QSA attention and PLE paths add dispatches the figure does not
carry. Below, the GPU term is a band: eta = 0.90 (Qwen3.6's measured end) and 0.75 [assumption: a guess for the unwritten kernels].

Resident set if the experts page: D + token_embd = 5.50 GB, plus ~1.5 GB of KV, scratch and runtime [assumption; the design doc's
room table reserves 1 GB for a general model, and 0.5 GB is 3.3 slots a layer, 2.5 % of S 131 [derived]]; PLE rows and cold experts
stay on the drive. PLE applies at layer 1 only (`qwen4exp.ple.layers = [1]`, header dump `q38-1.txt:29`), so a token needs one
more cut in the stream, after layer 0. Layer 0's GPU time is the window for reading the token's 16 PLE rows (1.4 KB): 0.60 ms on
the M4 Pro and 0.92 ms on the M1 Pro (the GPU row of 3.3 over 48 layers [derived]). The chain cost is ~0 if the reads end inside
it [unverified: a 16-row cold read has not been timed].

### 3.2 Hit rate: what the CUDA tier measured, and a replay on its traces

Measured on the box (`tierengine.md:140-160,471-486`): the gate plan's paged arena at S 154 slots a layer pays **45.7 fills a token**
(4,385 fills / 96 steps; 1.0-1.1 ms each) and the step is **the sum**: 26 ms card + 11.3 ms host + 46-50 ms fills = 83-95 ms
("measured untimed, r4b2"). Cold first-touch, S 141: 54 fills a token. At the r4b warm window (keep set) fills a token are ~0-1.
Those shapes mix card, host and arena tiers, so they do not give a hit rate against slots a layer.

A direct curve: a per-layer LRU over the maintainers' Qwen3.8 route traces (`resgen-long-q38`: six
3.3-4.7K-token prompts, three genres, 256 decode positions each; `q38-chat-d2`: 48 short chat requests, 256 decode positions
each; probe `py/lru_replay.py`, the arena carried from request to request, prompt rows update it, only decode rows are scored, a
position's own ids are never victims). [measured on those traces, `logs/lru-*.txt`]

    slots a layer S    64    96   128   160   192   224   256   320   384
    MB of arena       9.6  14.4  19.3  24.1  28.9  33.7  38.6  48.2  57.8   (x 48 layers x 3.134 MB)
    h, long prompts  68.8  77.2  83.1  87.3  90.4  92.8  94.6  97.1  98.7  %
    h, chat          74.6  82.3  87.3  90.7  93.0  94.7  95.9  97.8  98.9  %
    misses a token (long)  149.8 109.4 81.3 61.0 46.1 34.8 26.1 13.9 6.1

It does not reproduce `residency-general.md:377,392`. Its 88.4 % at 261-262 card slots a layer is the engine's card rule on another
trace (8 calls), not an LRU; this replay's LRU reads 94.6-95.9 % at S 256. The "per-layer LRU-64 ~67 %, LRU-128 ~81 %" quoted at
:377 from llama.cpp #27861 sit 2-8 points under this replay's 68.8-74.6 % and 83.1-87.3 % [derived from the table above].
Limits: 256 decode steps a request; a long decode leaves the keep set and loses hits (`tierengine.md:176-178` expects 10-25 fills a
token after 1,024 steps on its shape) [unverified for Mac]; the first 16 steps are 6.5-7.0 points lower than the whole on the long
prompts at S 160-256 (5.2 at S 128) and 1.6-2.6 lower on chat at S 128-256 [derived from the replay's `h_first16` column].

For V4.1-Flash (`py/lru_corpus.py`, 50,000-token prefill-ordered corpus sets, 38 layers x top 6, first 2,048 positions warm the
arena): S 64: 79-80 %; 96: 87-89; 118: 90-92; 128: 91-93; 192: 96-97; 256: 99; 320: 99.7 % [measured on prefill order, not
decode order]. The decode order is lower: the maintainers' V4.1 decode trace (`resgen-long-v41`, 40 layers
x top 6, the same per-layer LRU, decode rows scored; the 2026-10-25 re-review's replay) reads **S 147: h 90.7 %, 22.3 misses a
token; S 307: h 98.9 %, 2.7 misses** [measured on that trace]. Section 3.4 uses these, not the prefill-order 92.8-94.2 % and
99.5-99.7 %.

### 3.3 The options, one resource timeline each (per token, Qwen3.8, ms of busy time per unit; PCIe is absent on a Mac)

Unit demands used: GPU = 6.333 GB / (eta x measured stream rate), eta 0.90 and 0.75 (3.1); SSD = the fill time of each (token, layer)'s
k misses, summed: 3.134 MB a miss at the 1 MiB-part rate of queue depth min(3k, 4 fill threads) read off the table of 1.3 (linear
between QD 1, 4 and 16), weighted by the replay's per-(token, layer) miss histogram of 3.2. That gives 0.505-0.533 ms a miss in
the S 131 and S 220 columns and 0.493-0.503 at S 60 (the flat 0.50 used before the 2026-10-25 re-review moved the four sums
below by 0.2-0.9 ms); 16 fill threads change every figure below by at most 0.2 tok/s [derived from 1.3 and 3.2]. CPU = 48 x ~21 us (poll + commit + books) + 3 preads a miss (~10 us
each) [derived from 1.5]; DRAM = the GPU's bytes + the fills' write and read-back.

**(a) everything resident.** Qwen3.8 needs 111.3 GB, or 82.5 GB with the PLE table on the drive: the limit holds it only above ~128 GB of
RAM (limit 0.78 x 128 GiB = 107 GB, [unverified] fraction). GPU 6.333 GB at eta x BW; SSD, CPU ~0; wall = GPU. Needs no new
code beyond the PLE row reader (engram). On 32/48 GB it does not fit: 111.3 GB against 26.8/40.2 GB.

**(b) routed experts paged from the SSD into a resident arena (the CUDA nvtier's shape).** The columns below are the ceiling: S = 131
(M1 Pro 32 GB, arena 19.8 GB = limit - 5.5 resident - 1.5) or 220 (M4 Pro 48 GB, 33.2 GB) put the GPU set exactly at the limit
(26.80 / 40.20 GB), a state no probe has seen (1.2: the lead Mac stopped at 15 GiB), so these are the high ends of a band; the low
end is given after the table.

    unit (ms a token)      M1 Pro 32 GB, S 131, long | chat    M4 Pro 48 GB, S 220, long | chat
    GPU SMs + DRAM, eta 0.90 / 0.75    44.0 / 52.8              28.7 / 34.5
    misses a token (replay)   79.2 | 59.0                       36.0 | 26.5
    SSD (0.505-0.533 ms a miss)    39.9 | 30.3                   18.8 | 14.1
    host CPU: sync 1.0 + syscalls     1.0 + 2.4         | 1.0 + 1.8   1.0 + 1.1        | 1.0 + 0.8   (3 preads a miss x 10 us [assumption, tierengine.md:315]; the syscalls overlap the SSD wait)
    host DRAM (NVMe DMA)    0.25 GB = ~1 ms of bus, under the GPU's stream   0.11 GB
    PCIe                    0                                    0
    wall, sum (today's flow: miss -> stall), eta 0.90   84.8 = 11.8 tok/s | 75.3 = 13.3     48.6 = 20.6 tok/s | 43.8 = 22.8
    wall, sum, eta 0.75                                93.6 = 10.7 | 84.1 = 11.9           54.3 = 18.4 | 49.5 = 20.2
    wall, hits and shared expert under the miss's read in the layer, eta 0.90   78.2 = 12.8 | 69.6 = 14.4     45.7 = 21.9 | 41.4 = 24.2
    wall, lookahead window bound (recall 1), eta 0.90   69.3 = 14.4 | 62.3 = 16.1           42.2 = 23.7 | 38.6 = 25.9
    wall, SSD fully hidden (bound), eta 0.90            45.0 = 22.2                           29.7 = 33.6

The SSD, sum, in-layer and lookahead rows are the 2026-10-25 re-review's recomputation (its own script, not `py/predict.py`; inputs:
the miss histograms of 3.2, 1.3's QD table, the byte terms of 3.1, the GPU rates 160 and 245 GB/s of 1.4 and 1.6). The eta 0.75 rows
add 6.333 GB / (0.75 x BW) - 6.333 GB / (0.90 x BW) = 8.8 ms (M1 Pro) and 5.7 ms (M4 Pro) to the eta 0.90 sums [derived]. Demands next to the step (M1 Pro, long): GPU 44.0 + SSD 39.9 + sync 1.0 = ~84.8 ms,
the step itself: it is the sum of its units' demands with none hidden, not latency the resident warps cover. The wall is the
**sum** today: the router's ids exist only after the layer's attention, so a miss has nothing to hide under until the layer's own
hits and shared expert are put under it (lever 1); the CUDA box measured the same sum (83-95 ms = 26 + 11 + 48).

The low end, a Mac in daily use. The lead Mac stopped at 15 GiB of GPU buffers (1.2), so the arena is 15 GiB - 5.50 - 1.5 = 9.1 GB,
S 60 [derived: 16.1 GB - 7.0 GB; 9.1 GB / (48 x 3.134 MB) = 60.5]. M1 Pro, S 60, long | chat: 156.5 | 128.4 misses, SSD 77.2 |
63.7 ms, sum 122.1 = 8.2 | 108.6 = 9.2 tok/s; with lever 1 114.7 = 8.7 | 101.4 = 9.9; at eta 0.75 130.9 = 7.6 | 117.4 = 8.5; at the
lookahead bound 101.9 = 9.8 | 90.1 = 11.1 [derived from the same recomputation]. The M4 Pro has no stop measured, so its low end
is [unverified]. A lever on either unit predicts a real gain until they overlap; once overlapped the smaller (SSD) is the one under
the shadow and shortening it predicts 0. Flow levers, in order (asynchrony before shortening any term):
1. *Asynchrony inside the layer* (no recall term): after the router, the layer's hit experts and the shared expert do not depend on
   the misses, so their compute runs under the misses' reads. The CUDA design already counts it (`tierengine.md:163-164`: "the hits'
   ... compute under the miss's read", each expert's output stored and summed in list order to keep the bits). It hides 6.6 | 5.7 ms
   (M1 Pro, S 131) and 2.9 | 2.4 ms (M4 Pro, S 220) a token [derived: per layer, min(the fill time of its k misses, the GPU time of
   its 10 - k hits and the shared expert), summed over the histogram]: the in-layer rows above.
2. *Asynchrony across layers*: a gate lookahead (apply layer l+1's router to layer l's output; `tierengine.md:131-148`) names l+1's
   misses one attention sublayer early; the SSD runs under the GPU (measured: 98-100 % SSD and 94-97 % GPU together, 1.4). The
   window is one pre-router sublayer: 0.472 ms a layer on the M1 Pro, 0.308 ms on the M4 Pro [derived: attention, SSM, indexer and
   half the hyper-connection bytes over the layer's rate; the half is an assumption]. On these machines the window, not the recall
   r, decides: at r = 1 it hides 15.5 | 13.0 ms of the M1 Pro's SSD time (39 | 43 %) and 6.4 | 5.2 ms of the M4 Pro's (34 | 37 %);
   at r = 0.7, 14.8 | 12.3 and 6.4 | 5.2 ms. r (expected 0.6-0.85 [assumption]) is the CUDA design's own open term and a trace cannot
   give it (`tierengine.md:172-173`). It comes after lever 1: lever 1 needs no predictor.
3. *Bulk*: tokens share expert reads. A 2-4 token verify (MTP; a paged plan runs with the draft off today, README) reads the union:
   fewer than 2-4x the misses.
4. *SIMD/SIMT*: nothing new; the GPU side is already a stream at ~91-96 % of the read rate (Qwen3.6; 3.1).

**(c) host-CPU experts in unified memory (a NEON `qdot`).** Only the experts the arena holds are in memory, so the CPU adds
bandwidth, not capacity. The shared bus gives GPU 124-127 + CPU 46-75 GB/s together, 1.08x (4 threads) to 1.24x (8 threads) the
160.4 GB/s the GPU reads alone (1.6): at best -7..-19 % on the GPU-bound term (44.0 -> 40.9-35.5 ms on the M1 Pro), additive in the
sum model (3.1-8.5 ms of 84.8, +4..+11 % tok/s; the +35 % end of 1.6, which mixes baselines, would be 11.4 ms, +16 %) and worth 0
once the SSD sets the step [derived]. A mixed leg that computes a missed expert on the CPU saves only that expert's GPU time, 0.022 ms
[derived: 3.134 MB / (0.9 x 160 GB/s)], against a 0.5-0.6 ms fill that must happen first: at most 4 % of a miss, so it is rejected.
The CPU leg needs NEON kernels for Q4_K, Q5_K, Q5_1, Q8_0, IQ4_NL (the types the experts use) beside the GPU's, bit-exact or a new
accuracy pin. Predicted +4..+11 % at best, large cost: not first.

**(d) mmap'd weights left to the OS pager.** The GPU cannot fault: the driver pages in the whole bound range first (1.6: 2.0 GB/s
for 1 GiB, 16.4 ms vs 4.95 ms for one layer's 10 cold experts = 3.3x). A layer would also wire what it binds, so the OS cannot
evict it meanwhile. Predicted sum-model tok/s = (b) with c_miss x 3.3: M4 S 220: 28.7 + 36 x 1.65 + 1 = 89 ms = 11 tok/s [derived].
It is the cheapest to write and the slowest; useful only as a fallback when the tier refuses a load.

### 3.4 V4.1-Flash Q3_K_M (347.3 GB) on 128 and 256 GB Macs [derived; GPU rates are Apple's spec x 0.9, limits 0.78 x RAM, SSD 6.7: all [unverified] for these machines]

Header facts (`rig-log docs/bloomery/facts.md:7-16`, `crates/engram/src/lib.rs:11-14`, `tools/ref/router-residency.py:173`): routed experts
40 layers x 384 x 16.77 MB = 257.6 GB, top 6; engram tables 84.6 GB (110 B rows, on the drive by rows); dense 5.1 GB by subtraction.
A token: 5.1 GB + 240 experts x 16.77 MB = 4.03 GB = 9.1 GB.

    128 GB Mac (ceiling: limit 107 GB, arena 99 GB = 147 slots a layer): decode-trace h 90.7 %, 22.3 misses a token x 2.50 ms (16.77 MB
                at 6.7 GB/s) = 55.8 ms; GPU 18.5 ms (546 GB/s class); sync 1.0 -> sum 75.3 ms = 13.3 tok/s
    256 GB Mac (ceiling: limit 214 GB, arena 206 GB = 307 slots): h 98.9 %, 2.7 misses a token = 6.8 ms; GPU 12.3 ms (819 GB/s class);
                sync 1.0 -> sum 20.1 ms = 49.9 tok/s
    512 GB Mac: limit ~430 GB holds the whole 347 GB file: option (a), engram rows from the drive.

The hit rates are the decode trace's (3.2); the earlier rows used the prefill-order curve (92.8-94.2 % and 99.5-99.7 %) and read 16-18
and 61-65 tok/s with no sync term [derived: the same arithmetic]. Like 3.3's columns these sit at the limit, so they are the high end
of a band. On the 128 GB Mac the SSD term (55.8 ms) is 3.0x the GPU's (18.5 ms) [derived], and a flow lever's window is at most one
layer's GPU time, 18.5 / 40 = 0.46 ms, against a 2.5 ms fill: asynchrony cannot hide a miss there, so the arena size and bulk
(a verify batch sharing reads) come first. These rows rest on three unmeasured machine numbers; contributors' machines (#593 in the
census) can replace them.

### 3.5 Recommendation

**Design (b), memory kind `Pool`, on one common tier core.** The arena is one or more shared `MTLBuffer`s (each at most 0.585 x
RAM); the tier's books, LRU/heat and fill reader are the CUDA nvtier's, lifted out of `crates/gpu`; the PLE/engram rows keep the
`engram` readers. The Metal expert kernels take a slot table instead of expert ids (below). A paged token is cut at each layer's
router and nowhere else: a model of L layers is L + 1 command buffers, not two a layer. A buffer ends with the router, encoded
ahead and committed in order; the host polls the router's id words, claims and fills the misses with `pread`, writes the slot
table, then commits the next buffer, which holds that layer's experts and the next layer's attention up to its router. The
boundary between attention and experts inside a layer is not a handoff, and a cut there would only add a stream drain and a
boundary wait: the buffers' resources are untracked, so each buffer opens by waiting for the boundary event
(`crates/metal/src/stream.rs:14-20,187-195`). A layer with no misses costs ~21 us of host time.

*Common owner it reuses:* `NvTier` (books, reader, `TierSlots`), `engram::direct` (already macOS-aware), `crates/placement`
(already plans integrated pools; `Hosting::Unified`), the Metal residency set. *Per-Mac code:* the `Shared` arena kind, the slot
indirection in `experts.metal`, the poll/commit handoff in the stream. Not built: the card mover, the handoff page, tier nodes.

**The slot-table signature, fixed now so that the Qwen3.6 UD lanes of the design doc's 1c are written to it and `experts.metal` is
touched once.** Today the pair and fold kernels read `device const int * ids` (`ROUTED_IDS` = 8 entries,
`crates/metal/src/kernels/experts_geom.rs:2-5`) and address `((id * rows + row) * per_slot + local)` blocks from the tensor's
stack (`experts.metal:168-182`); slot 8 reads the shared expert (`SHARED_SLOT`, `experts_geom.rs:7-8`, compared at
`experts.metal:106,176`). Every expert lane, 1c's included, takes `device const int * slots` in the same place (same type, same
count, same shared-slot rule) and a per-tensor arena base in place of the stack. The address formula does not change: an arena
slot holds one expert's rows of that tensor in the stack's own layout, and a miss's fill writes its gate, up and down parts (the
3 parts a fill of 3.3) into the three arenas. `FoldArgs.n_experts` (`experts.metal:84-91`, 16 bytes, `static_assert`) is read as
the number of slots, the bound of the range check in `pair_expert` and `fold_ids_ok` (`experts.metal:103-127`): an entry out of
range raises the expert-id fault and NaNs the rows, which is the poison the proof below uses. Identity today means the arena is
the full stack and `slots[j] = ids[j]`, the router kernel writing the table; in the tier the host writes it from the polled ids.
k = 10 (Qwen3.8) is outside this signature: `EXPERT_SLOTS` 9, `ROUTED_IDS` 8 and `SHARED_SLOT` 8 are constants of
`experts_geom.rs` (`crates/metal/build.rs:118-120` hands them to the MSL), and the design doc's 1b-0 instance mechanism is where
they become a shape row's. `experts.metal`, `arena.rs` and `session.rs` have one owner at a time.

**Order: model breadth first.** The design doc's Phase 1 (1-0 kin, 1a instruments, 1c Qwen3.6 UD-Q4_K_XL, 1b) comes before the
identity round, for three reasons. The first real offload customer is Qwen3.6 (Q4_K_M 21.17 GB, UD-Q4_K_XL 22.35 GB, the design
doc's 1.1) on a 16 or 24 GB Mac, whose limit is 13.4 or 20.1 GB [derived: 0.78 x RAM; the fraction is [unverified] below 32 GB],
and the UD file needs 1c's Q8_0 and Q5_K lanes. Qwen3.8 and V4.1 are 15th and 17th in the design doc's cover order and have no Metal
kernels, so a "Qwen3.8 on the M4" step has nothing to run yet and the rows of 3.3 and 3.4 stay paper until then. And the identity
round's A/B base moves when 1b-1 changes attention (head-128 GQA), so it is measured once, after breadth. The Qwen3.6 rows (its
hit curve and tok/s on 16/24 GB) are the first paper step of the tier round: no Qwen3.6 route trace is replayed here.

**First round, closes the one term derivation cannot: what cutting a token costs ("Metal leg, identity map").** Scope: `crates/metal`
only, behind a same-binary lever that is a row of `crates/levers/src/registry.rs` (parsed in `main`, as `AGENTS.md` requires). It
cuts each token of the existing `qwen3moe` session at the router only: 41 command buffers a token on Qwen3.6 (40 layers). Each cut
ends at the router dispatch; the host polls the router's id words (with a per-layer sequence value, so that a previous layer's
equal ids cannot satisfy the poll [design]), writes the slot table from the ids it polled, and commits the next pre-encoded
buffer. No tier, no SSD. Proof: the table is written only from the polled ids, slot = id (the identity map), and every entry the
host did not write holds poison, an id past the stack, which `pair_expert` and `fold_ids_ok` take to the expert-id fault. A stale
or early poll writes the previous layer's ids or none, so the layer's rows differ from the lever-off arm's or the fault word is
raised: a stale read is red. With the poll right, tokens and the logits row are bit for bit the lever-off arm's (same kernels; the
cut changes no float order), on the M4's Qwen3.6-35B-A3B Q4_K_M.

Prediction [derived]: **+0.1..+0.6 ms a token**, i.e. 85.6 -> 84.9..81.4 tok/s at depth ~0 (11.68 -> 11.78..12.28 ms; -0.9..-4.9 %).
The point value is 40 cuts x (B 1.6 + C 1.5 us, the M4 probes of 1.5) = +0.12 ms, 84.7 tok/s (-1.1 %); C's loop already holds its
own commit-to-seen leg, so the sum may count it twice, and the floor is set at 0.1. The top of the band is the earlier estimate, 40 x
15 us = 0.6 ms (a poll, a commit and ~9 us of books and encode a cut), which put pre-encoded work on the chain. The unknown term is
what a cut adds beyond the probes: the boundary event the next buffer waits for and the host's drain of the cut buffer
(`stream.rs:187-195`; `MetalStream::settle`, `stream.rs:534`), and whether the one-deep run-ahead (`session.rs:332-336`) survives
the host polling inside a token [unverified]. The probes did not include them; the A/B measures them.

A/B: same binary, lever-on and lever-off arms plus an A/A arm (lever off against itself), 6 rounds on the M4. The ruler is the M4's
own, not the box's 0.6 %: sd 0.42 % over 3 rounds (85.5, 85.3, 86.0; rig-log `docs/bloomery/mac-m4pro-xeng-2026-10-25.md`), which
gives ±0.54 % at 6 rounds (t(0.975, 10) x 0.42 % x sqrt(2/6), `AGENTS.md`, "Know the ruler") [derived]. The A/A arm replaces that
3-round sd with the sitting's own, the way the Spark's ruler comes from its A/A (`docs/contrib/dgx-spark.md:195-200`); the card's
`sd:` line names the xeng record as its source. The band's floor, -0.9 %, is 1.6x the interval; the point value, -1.1 %, 2x. Outcomes:
- inside +0.1..+0.6 ms: the cut is priced; go on to the tier rounds with 40 x this cost as their chain term.
- below +0.1 ms: no ruler resolves it (0.1 ms is 0.86 % against ±0.54 %; ±0.3 % takes 32 rounds, `AGENTS.md`). Decide by
  derivation, not by more rounds: take the derived 0.12 ms (`AGENTS.md`'s "launch count only" row, dt = dN x the per-cut cost, 3.1 us in
  place of c_node) and go on.
- above +0.6 ms: the boundary event or the drain costs more than the probes saw. Price the cut with a doubling probe (a detached
  worktree whose one-line patch cuts every layer twice; the slowdown is the cost a cut, `AGENTS.md`) before any tier work, and
  re-derive the tier rounds' sync term (1.0 ms a token in 3.3) from it.

The slot table alone is not an arm: it replaces a 4-byte id load by a 4-byte slot load with the same address math, on kernels that
stream at 91-96 % of the read rate (3.1), which the model row of `AGENTS.md` for a bandwidth-bound kernel's instruction count
prices at 0.

Order after it (each with its own prediction):
2. The host-room sizing rule, before any tier code. The arena is sized from the host's room, not the census: `census.rs:40`
   counts the other processes as zero and breaks `devices.rs:33-36`. The rule is min(limit, the host's room at open) - resident -
   reserve, the room from `os::host_memory()` (`crates/os/src/lib.rs:507-552`: free + inactive pages), read at open and again
   before the first fill. Whether it predicts the 15 GiB stop of 1.2 is [unverified] and is the rule's first check.
3. The tier core becomes device-free (a move: `just canon-check BASE` identical for the gpu bins). Most of the ~3,200 test lines at
   `nvtier.rs:3779-7047` run on the Mac; the tests at `:5524`, `:6138` and `:6432` use `cuda_core` and stay on the box or move
   behind the memory-kind split.
4. The paging leg: `Shared` arena, forced small arena on the Qwen3.6 file (3.6); then Qwen3.8 on the M4 once its Metal kernels exist
   (predicted 18.4-22.8 tok/s at the ceiling, sum, eta 0.75-0.90; 3.3).
5. The in-layer hit/miss overlap (lever 1 of 3.3), before any lookahead: no predictor, -5.7..-6.6 ms on the M1 Pro and -2.4..-2.9 ms
   on the M4 Pro (12.8-14.4 and 21.9-24.2 tok/s at the ceiling).
6. The lookahead: r measured first by an engine record, as `tierengine.md:171-181` plans. Its bound is the window: 34-43 % of the
   SSD time at r = 1.
7. Prefill. A ubatch touches ~all 512 experts of a layer, so a paged prompt streams what the arena does not hold: at least
   (1 - S/512) x 77.0 GB = 43.9 GB (M4 Pro, S 220) or 57.3 GB (M1 Pro, S 131) a ubatch. The CUDA claim policy keeps S - 2B ids
   (B = 16) as Keep and reads the rest as Transient (`crates/model/src/moe.rs:1753-1770`), so a sweep re-reads 512 - (S - 32)
   experts a layer: 48.7 and 62.1 GB, 11 % and 8 % over the bound [derived]. SSD-bound at 6.7 GB/s, compute not in the bound.
   Time to first token for a 4,096-token prompt, from the bound's sweeps (6.56 s M4 Pro, 8.55 s M1 Pro at S 131, 10.15 s at S 60)
   [derived: ceil(4096/U) sweeps]:

       U = 512 (8 sweeps)        52.4 s = 78 tok/s (M4 Pro, S 220)   68.4 s = 60 (M1 Pro, S 131)   81.2 s = 50 (M1 Pro, S 60)
       U = 2,048 (2 sweeps)      13.1 s = 312                        17.1 s = 239                    20.3 s = 202
       U = P = 4,096 (1 sweep)    6.6 s = 625                         8.6 s = 479                    10.1 s = 404

   A ubatch of U = P needs an arena that holds 4,096 tokens' activations [unverified], and U comes from that arena, not from the
   tier. In a U = P sweep, layer l+1's reads (0.92 GB, 0.14 s on the M4 Pro [derived: 43.9 GB / 48 layers / 6.7 GB/s]) go under
   layer l's compute; whether a 4,096-token layer's compute covers them is [unverified] (not derived here).

### 3.6 How it is shown correct without the real file

The CUDA gate does the right comparison but only on the real file: `gate_nvtier` runs a paged arm against an all-RAM arm, every
greedy token equal and the prompt's and each step's logits row bit for bit (`gate_nvtier.rs:6-10`, clause `bits`); `nvtier_family.rs` does it for
V4.1 and GLM; clause `refuse` (`gate_nvtier.rs:21-33`) loads at exactly the floor's slots and serves a prompt. The paged room is forced through
`BLOOMERY_HOST_ROOM` and `BLOOMERY_NVTIER_BYTES` (`crates/levers/src/registry.rs:47`). It has no fixture arm: `justfile:2758` is
`real-only`. For the Mac, force the arena on a file the M4 holds: Qwen3.6-35B-A3B Q4_K_M (21.2 GB, 40 layers x 256
experts, top 8; 1.3). the second M1 Pro read Qwen3-30B-A3B instead, so the same file on the other Macs is [unverified]. The slot count sits at the floor (`slots_of`, `nvtier.rs:365-390`: the larger of k x columns + lane slots and 2 x
`UNION_BATCH_IDS` + 5, about 37 for k 8). The paged arm must equal the resident arm bit for bit because the arena holds the
file's bytes and the kernels are unchanged. FAIL-first: zero a fill, swap two slot-table entries, and the first round's stale-poll
mutation (the table written from the previous layer's ids). Counters: misses, fills and evictions all > 0. The tier core also has a
counting identity independent of any model: replay a route trace through `NvTier` and compare hits and fills with `py/lru_replay.py`'s,
no card, no model file. The replay must feed the tier step claims only (the replay variant where prompt rows do not touch the arena):
`NvTier` claims a prompt call by column count as Keep and Transient (`crates/model/src/moe.rs:1753-1770`), which the LRU replay does
not model. That policy moves h by at most 1.3 points [measured: the 2026-10-25 re-review's replay with the prompt rows as LRU touches,
not at all, and as the claim policy], so it is not the same policy and the identity test carries no prompt claims.

On Qwen3.8 the Macs have no all-RAM arm (111.3 GB against 26.8 and 40.2 GB). The claim becomes arena-size invariance: the same
prompt and steps with the arena at the floor and at the most the room gives, both paged, every greedy token equal and the logits rows
bit for bit. It adds the `audit` clause of `gate_nvtier` (`gate_nvtier.rs:17-20`), which `memcmp`s each pick's slot against a fresh
buffered read of the same file range. What stays unproven after both, by name:
- the Q5_K, Q5_1 and Q8_0 slot kernels (the floor proof covers the types the Q4_K_M file's experts use; read its header);
- k = 10: `SHARED_SLOT` is the constant 8 (`experts_geom.rs:8`, compared at `experts.metal:106,176`) and Qwen3.6 is k = 8;
- PLE row reads and fills sharing the one drive.

### 3.7 Open questions

1. Past the limit (section 1.2) needs an idle Mac; the machine in use is the realistic case and stops 10 GiB early. Every high end
   in 3.3 and 3.4 stands on it, and the M4 Pro's low end has no stop measured.
2. Working-set fraction, GPU rate and SSD rate of 16, 24 and 64-512 GB Macs (the Qwen3.6 first customer and the V4.1 rows).
3. Hit rate over a decode longer than 256 steps, on a Mac user's prompts, and V4.1 in decode order.
4. Lookahead recall r for Qwen3.8 and whether the router of layer l+1 can run before its attention in the Metal stream.
5. Whether `F_NOCACHE` at 4 KiB skew (the tier's `DIRECT_ALIGN`) costs against 16 KiB alignment.
6. The cost of 41 command buffers a token, cut at the router, to the real encoder (the first round): the boundary event and the
   drain per cut, and the run-ahead.
7. Sustained SSD + GPU on a laptop (power, thermals); 6.7 GB/s was measured in bursts of 0.1-0.7 s.
8. The user-side `iogpu.wired_limit_mb` (read only here, never set): +2-3 GB of arena on a 32 GB Mac is +13 % slots, h +1-2 points.
9. Qwen3.6's own route trace: its hit curve and tok/s rows on 16/24 GB Macs, and the Qwen3.8 GPU term (eta 0.75-0.90) once its
   Metal kernels exist.
10. Whether the host-room rule (3.5, order 2) predicts the 15 GiB stop of 1.2.

## 4. Commands (probe sources under the scratch dir; ssh with `-o BatchMode=yes`)

    swiftc -O src/limits.swift -o bin/limits && ./bin/limits                          # 1.1 (copied to the other two Macs)
    ./bin/pastlimit                                                                   # 1.2, lead Mac
    ioprobe FILE 1024 all | 4096 read | 2048 all      # 1.3, the second M1 Pro and the M4 Pro
    overlap FILE overlap 1 ; overlap FILE mmapbuf 1 ; overlap FILE mmapexp ; overlap /dev/null cpugpu 1    # 1.4, 1.6
    coherence FILE 500 ; latency ; spin                                               # 1.4, 1.5
    python3 -I py/lru_replay.py TRACES/{resgen-long-q38,q38-chat-d2} 32,64,...,448          # 3.2
    python3 -I py/lru_corpus.py TRACES/{prose,code,korean} 64,...,320 --k 6 --layers 2:40 --skip 2048
    python3 -I py/q38bytes.py q38-{2,3,4}.txt ; python3 -I py/active_bytes.py 256 8 q36-ud.txt ; python3 -I py/predict.py
    # the 2026-10-25 re-review recomputed 3.3, 3.4 and 3.5 with its own scripts over these logs and the decode traces (not in the repo)

## Re-review (2026-10-25)

Each finding of the adversarial re-review, checked against the code before it was applied; none was disproved by a code line.

- S1-1 applied §3.5 (first round cut at the router only, 41 buffers; +0.1..+0.6 ms; M4 A/A arm and sd; branch below the floor).
- S2-1 applied §3.5 (slot table written from the polled ids, the rest poison, `experts.metal:103-127` faults on it).
- S2-2 applied §0.5, §3.3 (the 70 % rows withdrawn; the window bound, 34-43 % at r = 1).
- S2-3 applied §3.3, §3.5 (in-layer hit/miss overlap, `tierengine.md:163-164`, ahead of the lookahead).
- S2-4 applied §3.2, §3.4 (V4.1 rows from the decode trace: 13.3 and 49.9 tok/s).
- S2-5 applied §0.5, §3.3, §3.5; the M4 Pro's low end stays [unverified], deferred: a past-limit probe on the M4 Pro.
- S2-6 applied §3.5 (4,096-token TTFT, the U = P sweep, the CUDA claim policy's bytes).
- S2-7 applied §3.1, §3.3 (eta 0.75-0.90 for Qwen3.8).
- S2-8 applied §3.6 (arena-size invariance, the `audit` clause, three unproven items; the fixed slot 8 is `experts_geom.rs:8`).
- S2-9 applied §3.5 (breadth first, one owner at a time, Qwen3.6 on 16/24 GB first; 0.78 and the 1 GB / 1.5 GB split stated);
  deferred: weighting the design doc's tiers by what offload serves, and Qwen3.6's own hit curve and tok/s rows, to the tier round.
- S3-1 applied §1.5 (probe C answers it; timeouts 0).
- S3-2 applied §3.2 (first 16 steps 6.5-7.0 points lower on long prompts at S 160-256).
- S3-3 applied §3.2 (the 88.4 % is the engine's card rule on another trace, not an LRU).
- S3-4 applied §3.6 (not the same policy; step claims only, at most 1.3 points).
- S3-5 applied §0.6, §1.6, §3.3 (+8 / +24 % against one baseline, +4..+11 % at best; CPU-on-miss rejected at 0.022 ms of 0.5 ms).
- S3-6 applied §1.4 (147-160 GB/s).
- S3-7 applied §2 (`tierengine.md:283-284`, `workstation.rs:1164`, `census.rs:40`, `devices.rs:33-36`).
- S3-8 applied §3.5 (the lever is a row of `crates/levers/src/registry.rs`).
- S3-9 applied §3.5 (decided by derivation; the slot table alone is not an arm).
- S3-10 applied §3.5 (sweep bytes 48.7 / 62.1 GB under the claim policy).
- S3-11 applied §3.1 (PLE at layer 1 only, `q38-1.txt:29`; one more cut a token; window layer 0).
- S3-12 applied §3.6 (the M4 holds the Qwen3.6 file; the second M1 Pro read Qwen3-30B).
- S3-13 applied §3.5 (nvtier tests at `:5524`, `:6138`, `:6432` use `cuda_core`).
- S3-14 applied §3.3 (0.505-0.533 ms a miss from the histogram, sums +0.2..+0.9 ms).

Sources: the 2026-10-25 adversarial re-review of this document (findings S1-1 to S3-14, numbered here in its order) and, for the
order, the same-day re-review of the design document. The lead's decisions of that day set the order (model breadth first, the
offload identity round after) and the wording of the predictions above.
