# Hardware

bloomery is built for one kind of machine: one or two RTX 3090 cards, an AVX2 host with many memory channels, 256 GB of RAM and an NVMe drive. This page says what each part costs in a decode step. Every number carries its source; a number marked [derived] was computed, not measured.

## Target profile

| | T1, the minimum | T2, extended | The development machine |
|---|---|---|---|
| GPU | RTX 3090 24 GB ×1, placement `gate`: dense weights 7.66 GB plus 888 experts, 14.9 GB | RTX 3090 ×2, 48 GB. Needs a two-card path that is not built yet | RTX A6000 48 GB, placement (a): 2,414 routed experts, 40.5 GB, on the card |
| CPU | AVX2, 8 DDR4 channels | same | Threadripper PRO 5975WX, 32 cores, 8 × DDR4-3600 |
| RAM | 256 GB | same | 264 GB |
| Storage | NVMe, for the engram table | same | NVMe |

All timed numbers so far come from the development machine's A6000. T1 has not been timed yet. Its predicted rate is 22–24 tok/s without a draft [derived: fewer experts on the card means about 30 % more host experts, about 7–8 ms more per step].

## What each part costs

The decode step has three legs that overlap: the GPU kernels, the host expert leg, and the engram rows. The host leg is the longest.

**Host memory bandwidth is the step's dominant term.** Every token reads the routed experts that are not on the card from RAM. With placement (a) on the A6000 that is about five host experts per routed layer. A host-tier bench with exactly five per layer reads 3.37 GB per token at 135–137 GB/s with 32 threads, which is 24.89 ms per token (rig-log [2026-09-24](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)). The machine's own read ceiling is 147.7 GB/s at 32 threads (rig-log 2026-09-14, after the memory clock was raised to 3600). The tier saturates at 16 threads.

**The card waits on the host.** An nsys trace of a 41.7 ms step at depth 6 (A6000, prefix placement; the tracer itself lengthened the step by 1.5–2.5 ms) had 17.36 ms of kernels and a 28.49 ms host leg, of which 23.91 ms was an exposed wait for the host (rig-log [2026-09-24](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md)). Putting the most-used experts on the card (the hot list) cut the step from 40.0–40.2 ms to 35.2–35.4 ms (A6000, depth 6, `n = 96`, instrumentation off).

**The GPU kernels** are bandwidth-bound matrix-vector products. Large launches read at 567–701 GB/s on the A6000 (`docs/plan.md`, cost model).

**engram rows come from NVMe.** The table is about 195 GiB and is memory-mapped. One token reads 48 rows of 272 bytes. Read cold as page faults, they took 4.69 ms (median); with each row's read-ahead issued one token early, 0.31 ms (rig-log [2026-09-22](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-p-the-engram-path-costs-a-third-of-a-millisecond-and-it-is-syscalls.md)). On real text more rows are cold than on the synthetic depth prompt: a 4096-token prose prompt added 3.3 ms per step outside the host leg (A6000, depth 4096, prefix placement, instrumentation on; rig-log 2026-09-24).

## RAM floor

256 GB is the floor. The host expert set is 196–244 GB, depending on how many experts the card keeps (`docs/plan.md`, target profile). With placement (a), 12,946 experts, 218 GB, stay on the host as a memory-mapped file (rig-log 2026-09-23). The engram table does not fit beside them [derived: 218 GB plus the table's 209.2 GB is more than the RAM], which is why it is read from NVMe.

## A 4-channel board

A board with 4 DDR4 channels has about half the host bandwidth [derived]. The host leg then takes about twice as long, near 50 ms per token instead of 25 ms [derived]. Because the card already waits on the host, the step grows by about that difference [derived, not measured]. Nothing has been measured on a 4-channel machine.

## GPU architecture

Every kernel is built for sm_86 (Ampere: RTX 3090, RTX A6000) and gated there. Other architectures have not been built or tested.
