# r7g predictions (GLM-5.3-Flash on the NVMe tier), written before the code

G = GiB (`BLOOMERY_HOST_ROOM` takes binary units). Real file = UD-IQ4_XS, `--plan`, place a, the 3090's gate card.
Rooms: 128G = 137,438,953,472 B; 118G = 126,701,535,232 B; 108G = 115,964,116,992 B. On a unified machine (PR #16) the
room read is the pool and the plan's need holds the cards' share (`HostNeed.cards`), so the host's room is the pool less
that share.

## Terms, from the code and the printed line
- H = 143,690,563,584 B host experts (the printed line). Check by type: 288 experts x 42 layers, E x F = 2^23, IQ3_S 110 B,
  IQ4_XS 136 B, Q6_K 210 B a 256 weights: 38 layers (IQ3_S, IQ3_S, IQ4_XS) 3,359,637,504 B; 3 Q6_K-down layers
  (REAL_Q6K_DOWN = 11, 12, 44) +698,351,616 B each; one IQ4_XS gate/up layer +490,733,568 B: sums to H exactly.
- need N = 148,743,454,720 B, base = N - H = 5,052,891,136 B, W = 4,548,722,688 B (the design's; the layer that is
  IQ4_XS gate/up with a Q6_K down). H does not tell it from W = 4,057,989,120 B (that layer's down IQ4_XS): the
  `HostRoomFloor` text at a 10G room prints need, base and W, which settles it.
- floor F = base + 3W = 18,699,059,200 B; H/2 = 71,845,281,792 B: every room above is >= H/2, so the default arena is 0
  (S = 0, arena_slots 0).

## Predicted plan lines (paged = N + 2W - R + d, d in [0, 15.8 MB): the greedy one-expert fill leaves < one unit)
| room | paged | share of H | arena | S |
|---|---|---|---|---|
| 128G | 20,401,946,624 | 14.2 % | 0 | 0 |
| 118G | 31,139,364,864 | 21.7 % | 0 | 0 |
| 108G | 41,876,783,104 | 29.1 % | 0 | 0 |
If W = 4,057,989,120 every row is 981,467,136 lower. No room refuses (all above F and above F + 33 slot sets).

## NextN room term (`--plan` with the draft on, same rooms)
Trunk split at R - D, D = the draft layer's host experts: paged_nextn(R) = paged(R) + D while R - D >= F. D = 3,359,637,504 B
if the layer follows the 38-layer type map (band 3.36 to 4.55 GB). The plan record shows the trunk only; D = paged_nextn - paged.

## The iq4xs fixture (derived from `fixture.rs` and the type map; F and base are read off the probe like R7b's)
- 4 routed trunk layers (f2 to f5), ff 512: per expert 901,120 (IQ3_S gate) x 2 + 1,114,112 (IQ4_XS down) = 2,916,352 B;
  W = 288 x 2,916,352 = 839,909,376 B; H = 3,359,637,504 B; card experts 0 (no IQ3_S card rule), lanes 0.
- slot set (each stack in its window, `arena_window_bytes`) = 4 x 2,940,928 = 11,763,712 B; floor slots = max(8 + 0, 33) = 33;
  A at S = 64 = 752,877,568 B. Default dial: 3W = 2.52 GB > H/2 = 1.68 GB, no deep side, so arena 0 and the arena is given.
- At room F + A the split keeps W (keep = H - (N + 2W - F) = W), 72 ids a layer, pages 216 ids a layer: paged = 3W =
  2,519,728,128 B exactly, arena A, arena_slots 64, 4 paged layers, carve 64 < 216 served ids.
- Run (P = 512, one batch, then 8 greedy steps): fills ~ 864 for the call (every paged id of 4 layers), ~ 135 more in the
  steps (band 80 to 190); evictions ~ fills - 64 x 4 ~ 750 [650, 950]; unions > 0; mapping reads 0, slot reads 0. The clauses
  need only > 0.
- NextN draft (fixture keeps Q4_K gate/up, Q5_K down): D = 288 x (2 x 1,179,648 + 1,441,792) = 1,094,713,344 B > A. At the
  paged arm's room F + A the trunk gets F + A - D < F: `HostRoomFloor` by name (the NextN room refusal). At F_n + A + D
  (F_n read off the NextN plan's own probe) it plans, pages, arena A. Without the room term the first plans: the clause goes red.

## Resource timeline (per gate item; correctness gate, no timed term, no lease)
| Unit | Host CPU | Card SM | PCIe | Host DRAM | NVMe |
|---|---|---|---|---|---|
| fixture load (~8.6 GB file) | ~10 s | idle | ~5 GB upload ~1 s | file in page cache | ~8 GB cold read once |
| prompt call, paged arm | host leg over the union | ms | small | arena 0.75 GB filled | ~1 GB O_DIRECT, ~0.3 s |
| 8 steps | tens of ms | ms | small | -- | ~0.1 GB |
Wall = sum of two loads + build: 1 to 3 min a batch item. Flow lever: none (a correctness gate).
