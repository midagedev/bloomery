# refill: an evicted slot's refill stops giving its pages back first (predictions, before code)

## 1. Why give_back sits before the refill: no correctness reason
- R4b-2 (47297678): "Slot pages return only through give_back (integrated devices: design Q18)". Q18 §4: an arena slot's
  pages return "only through the one NvTier give-back ... R4b-2 adds no AnonMap::dontneed call": a one-door rule. The call
  before a pick's read is older: R2a's `claim` loop ran `dontneed` per evicted slot (47297678^ nvtier.rs:1873) for the module
  doc's RSS premise; R4b-2 moved it into `read_pick` unchanged.
- Q18 §2 wants the opposite: give_back "a no-op for Registered (the slot is overwritten by its next fill)", and "Dropped on
  Registered: the eviction's early page return". Q18 §1: DONTNEED on a device-registered slot makes the next fill write pages
  the device never sees (mutant D1). A refill that gives back needs a no-op branch there; one that never does needs none.
  Structural verdict: the refill calls it on neither kind of device; only `release` (a landing, nvtier.rs:1729) keeps it.
- Net RSS of an eviction is 0: `claim.evicted` slots are the picks with `gone`, refilled in the same fill (claim, :2090).

## 2. Safety of skipping it
- (a) Readers touch only [skew, skew+len) = [0, need) of a window: `TierSlots::slot` nvtier.rs:2511, `staged_slot` :1633
  (both `window_span` :540); the step's lend moe.rs:2154 with its length check :2173; the lane's copy swap_source.rs:738;
  `audit_slots` :2009 (`window_span` :2027, offsets over `r.len`). `range()` :1588 is an address pair (gate_nvtier.rs:1760).
- (b) A short read is refused by name: `got < part.need` read_pick :2249 (`GpuError::shape`); bytes past `got` (EOF) were zero,
  are stale now, and sit past `need`.
- (c) `resident_bytes` is the books' count (+slot_bytes a pick count_fill :2286; -slot_bytes per eviction :2093 and release
  :1725), not RSS; unchanged. The arena is full-size resident after warm-up either way.

## 3. The term and its timeline (r4b shape: A6000, plan a, prose 512, n 96, mid-p0-s1, 48 layers)
- Refills a token f = 15..30, central 22 [r4b-decode.card term 1: 49.8 base fills x (1 - c), c 0.40..0.70; anchor: the gate
  plan's 4,385 fills over 96 steps = 45.7 a token, measured]. Main evicts on every one (arena full).
- A layer's fills run on the pool in the chain (`fill_missing` -> `fill` -> `for_each_chunk`): parallel across a layer's picks,
  serial across layers. Layers with a miss L = 48 (1 - e^(-f/48)) = 12.9 / 17.6 / 22.3.
- Pick timeline, one thread, 3 sync reads of ~1.05 MB: dontneed 0.07..0.10 ms (zap + TLB IPIs) | GUP faults of 772 zero pages
  0.30..0.50 ms, before each bio is submitted, so not under the earlier bio's I/O | O_DIRECT 3.16 MB at 2.2..2.8 GB/s =
  1.13..1.44 ms. CPU s [paper] 0.4..0.6 ms; hint it is more: the lane's victim fill (same body) waited 2.41 ms vs 1.2..1.6
  predicted (r4da: 553 jobs, 1334.7 ms; decisions.md:59). I take s = 0.40..1.0, central 0.6.
- Per token on main [derived; card's T = 45..69, central 58 ms]: host compute ~26 ms (48 x 0.54; DRAM ~1.0 GB), fill wall
  ~32 ms (18 layers x ~1.8), card SMs ~7 ms and PCIe (17.9 MB) ~1 ms under the chain, NVMe 69.5 MB = 23 ms busy. The wall is a
  sum: a layer's compute waits for its fills, and at m = 1 the drive (4.6..6 GB/s) is not the bound (one pick pulls 2.2..2.8),
  so s is on the critical path. m >= 3 is 1.3 % of miss layers: kappa = 0.9..1.0.
- g = L kappa s = 4.6 / 10.0 / 22.3 ms; tok/s gain T/(T - g) - 1, T paired with f (high f, high T): +7 % (g 4.6, T 69) .. +49 %
  (g 22.3, T 68), central +21 % (g 10.0, T 58). Cross-check, gate plan: L = 29.4, g = 16.8 ms of its untimed 93.5 ms step.
- Prompt: the reader's 16 threads are drive-bound (5.4 GB/s), the page work sits under the drive: wall 0. Tokens and decode
  fills equal in both arms. The one term derivation cannot fix is s; the A/B measures it.
- Ruler: sd 2.6 %, N = 4: h = 4.50 %, edge +7 clears it by 2.5; N = 6: h = 3.34 %. docs/cards/refill-decode.card: `card.py check` ok.
