# sparkdec: the GB10 gemv read gap, and the two decode levers it was asked about

Design round, 2026-10-11, main `347a86e5`. No box run, no Spark run. Model: Qwen3.6-35B-A3B `Q4_K_M` whole on one
DGX Spark (GB10). Every number below is either the contributor's measurement (cited by comment), read from the code
(`path:line` at `347a86e5`), cited from a source (URL and one quoted line), or `[derived]` from those.

## Verdict

| Lever | GB10 step Δ (11.50 ms) | A6000 Δ | Decision |
|---|---|---|---|
| 1a. `qwen35moe_router_norm` regridded: 8 rows a block (one a warp), 33 blocks, the first weight chunk issued before the norm | −0.07 … −0.22 ms (−0.6 … −1.9 %), center −1.3 % `[derived]` | −0.06 … +0.02 ms (−1.0 … +0.4 %) `[derived]` | **Open** (round `sparkrouter`, §7.1): a kernel defect on 48 SMs (a second wave), not a PDL question. Its one-warp routing tail (2.5–4 µs a launch, 0.10–0.16 ms a step) stays; a bit-keeping rewrite of it is worth −0.02 … −0.04 ms, model says 0 |
| 1b. PDL across the step's 520 nodes | −0.20 … −0.67 ms (−1.7 … −5.8 %) `[derived]`, band too wide | 0 by construction (sm_86 PTX cannot carry it) | **Not yet**: needs the `sm_121` target (`archkey`), a fork patch (ledger row 42), and one term, GB10 `c_node` (§4.4) |
| 1c. `qwen3moe_o_resid_q4k` second wave (256 blocks on 240 slots) | −0.07 ms (−0.6 %) | 0 | **Model says 0, do not open** (under the ruler; PDL would fill it) |
| 1d. `q4k_gemv_sel` tail (half-warp at K = 512, a dependent id load) | ≤ −0.09 ms (−0.8 %) | 0 | **Model says 0, do not open** |
| 2. More bytes in flight per SM on the layer gemvs and the head | Little's-law form (more warps, blocks or loads to cover latency): 0. The same code shape for DRAM row locality (a row's loads issued together): up to −1.4 ms, undecided | 0 | **Little's-law form: model says 0, do not open** (§5.3). **Row-locality form: undecided until the probe's arm P** (§5.4), which also tests load width (arm W); the 1.63 ms the gemvs lose is that term |

The rule in `docs/contrib/dgx-spark.md:112-117` leaves 160–200 GB/s open. This reading closes it: of the 2.75 ms the
gemvs spend past the plain read rate, about 1.63 ms is a steady-rate deficit that every big gemv shares (the head
included, at 99.8 % SM active), about 0.67 ms is per-kernel tails, about 0.23 ms is the per-launch intercept,
mostly nsys's own, and the last ~0.2 ms is the small grid-8 `qkv_q4k` and grid-64 `q6k_gemv` launches (§2). Neither lever as asked reaches the 1.63 ms: in-flight bytes are not what it is short of (§5.3).

## 1. Facts used

From [#17 comment 6104914439](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104914439) and
[#17 comment 6104639179](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104639179) (build `04d0c0ae`,
crates equal to `a3a44ca2`; stock clocks unless noted):

- Untraced graph step p50 11.50 ms (11.55, 11.65, 11.65 over three processes), 2061.4 MB a step. At the plain read rate
  235.6 GB/s (#13's `hostread2`, `cuMemAlloc`, gate_up rows fresh) the bytes take 8.75 ms.
- Traced (nsys, eager) mean step 12.26 ms of kernel time: 8.40 gemv bytes at 235.6, 2.75 gemv time past that rate,
  1.12 other kernels (83 MB). The nsys sum is 1.6–3.6 % above the untraced eager step (11.90 ms): nsys adds ≥ 0.70 µs a launch.
- Per kernel (in-flow µs, MB a launch, launches a step): table in §2.
- Clock: capped 1989 MHz → stock 2450–2483 MHz moves the step 0.14–0.31 ms. ncu SM active (base clock): gate_up
  94.3 %, qkv 91.1 % (grid 8: 12.6 %), o_resid 87.4 %, q4k_gemv_sel 92.5 %, q6k_gemv_sel 93.3 %, router_norm 61.1 %.
- L2 read sectors from the SMs are 1.01–1.05 × the memory fills on the quantized layer gemvs: no duplicate L2 traffic.

Code facts (registers and static blocks per SM from the 2026-10-11 `ptx-scan` of `bloomery-serve`, ptxas 13.3, sm_86;
the cc 12.1 JIT may allocate differently `[unverified]`): `qwen3moe_gate_up_swiglu_q4k` 48 regs / 5, `qwen3moe_qkv_q4k`
64 / 4, `qwen3moe_o_resid_q4k` 48 / 5, `q4k_gemv_sel` 48 / 5, `q6k_gemv_sel` 40 / 6, `q6k_gemv` 40 / 6,
`qwen3moe_head_q6k_argmax` 40 / 6, `qwen35moe_router_norm` 47 regs, 9264 B smem / 5. Every weight load in these entries
is `ld.global.b32` (a 2026-10-08 PTX dump of the decode bin: 67, 133, 67, 21, 72, 20 and 66 scalar loads, no vector
load). Per-SM limits at cc 12.x equal cc 8.6's: 1536 threads, 48 warps, 64 K registers, 100 KB shared
([CUDA programming guide, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html);
the page's table groups 12.0 and 12.1 as "12.x" `[cited; the flattened table read by column order — confirm with
cuDeviceGetAttribute on the Spark]`). So the static blocks per SM carry over, and a grid's resident slots are
48 × blocks/SM on GB10 and 84 × blocks/SM on the A6000.

## 2. Paper decomposition of the step

Steady rate and intercept from the one kernel that runs at four sizes, `qwen3moe_qkv_q4k` (grids 520 and 1536):
slope (72.99 − 25.39) / (14.176 − 4.810) = 5.082 µs/MB, i.e. **R_ss = 196.8 GB/s**, intercept **t0 = 0.95 µs**
`[derived]` (grids 1088 and 1152 sit 1.5 and 1.2 µs above the line). The head, 2.1 ms long, gives its own steady rate,
199.2 GB/s. Each kernel's time is then `t0 + B / R_ss + tail`:

| Kernel (grid) | MB | n/step | in flow µs | at 235.6 µs | t0 + B/R_ss µs | tail µs | tail ms/step | waves GB10 / A6000 |
|---|---|---|---|---|---|---|---|---|
| `gate_up_swiglu_q4k` (576) | 10.652 | 40 | 57.97 | 45.21 | 55.08 | 2.89 | 0.116 | 2.40 / 1.37 |
| `head_q6k_argmax` (31040) | 417.268 | 1 | 2094.94 | 1771.1 | (R 199.2) | ~0 | 0 | 108 / 62 |
| `qkv_q4k` (1536) | 14.176 | 16 | 72.99 | 60.17 | 72.98 | 0.01 | 0 | 8.0 / 4.6 |
| `q6k_gemv` (1024) | 13.779 | 14 | 74.10 | 58.49 | 70.97 | 3.13 | 0.044 | 3.6 / 2.0 |
| `o_resid_q4k` (256) | 4.747 | 40 | 26.83 | 20.15 | 25.07 | 1.76 | 0.070 | **1.07** / 0.61 |
| `q6k_gemv_sel` (2304) | 7.758 | 20 | 44.10 | 32.93 | 40.37 | 3.73 | 0.075 | 8.0 / 4.6 |
| `q4k_gemv_sel` (2304) | 5.335 | 20 | 32.66 | 22.64 | 28.06 | 4.60 | 0.092 | 9.6 / 5.5 |
| `router_norm` (257) | 2.160 | 40 | 18.48 | 9.17 | 11.93 | 6.55 | **0.262** | **1.07** / 0.61 |
| `qkv_q4k` (520 / 1088 / 1152) | 4.8 / 10.0 / 10.6 | 14 / 6 / 4 | 25.39 / 53.47 / 56.18 | | | 0 / 1.47 / 1.18 | 0.014 | 2.7–6.0 |

Step sum `[derived]`: steady deficit Σ B·(1/R − 1/235.6) = 1.31 ms (layer gemvs at 196.8) + 0.32 ms (head at 199.2)
= **1.63 ms**; tails **0.67 ms** (router 0.26 of it); intercepts 237 × 0.95 = **0.23 ms**; the 0.2 ms left of the
contributor's 2.75 is the grid-8 `qkv_q4k` and grid-64 `q6k_gemv` launches, which the table's "15 other pairs" hold.

Clock share `[derived]`: step = A + C/f with Δ(1989 → 2466 MHz) = 0.14–0.31 ms gives C/f = 0.58–1.29 ms at stock, i.e.
5–11 % of the step scales with the SM clock. The rest (≥ 10.2 ms) is clock-free: bytes, DRAM-side latency, fixed ns.

Per-SM unit demand at the plateau, next to the step's cycles (AGENTS.md's q3gemma rule) `[derived]`: 196.8 GB/s over 48
SMs at 2.44 GHz is 1.7 B/cycle per SM. Q6_K head: ~120 warp-instructions per 420 B warp-iteration (15 loads, 12
funnels, 4 dequant/dp4a chains, scale) → 0.49 warp-instr/cycle per SM against 4 issue slots (12 %). L1TEX: ~5
wavefronts per scalar-load instruction → ≤ 0.25 wavefront/cycle against 1. No SM unit is near its demand; the step is
not SM-bound.

## 3. Resource timeline per decode step (GB10, graph mode, 11.50 ms)

| Resource | Busy | Basis |
|---|---|---|
| LPDDR5x (one pool for card and host) | 8.75 ms at 235.6 GB/s; delivered 2061.4 MB / 11.50 ms = 179 GB/s average | contributor bytes; #13 rate |
| Card SMs | warps resident most of the step (SM active 87–94 % on the gemvs), issue ≤ ~12 %, L1 ≤ 25 % | §2 |
| Host CPU | one graph launch and the sampler a token; off the critical path while the graph runs | `[unmeasured on the Spark]` |
| PCIe, NVMe | 0 (whole on card, integrated device) | `docs/contrib/dgx-spark.md` §2 |

Inside a kernel the wall is a **max** (bytes at the stream's rate against the warps' chain, overlapped). Across
kernels it is a **sum**: 520 dependent nodes on one stream, each paying its own ramp (first bytes one DRAM latency after
its blocks start) and its own tail (the last wave draining with few warps), plus the node gap. Flow levers, in the
order the timeline asks:
- **Asynchrony across the node boundary** (PDL): start node n+1's weight reads under node n's tail. This is the only
  lever that attacks the sum's 520 boundary terms at once (lever 1b).
- **SIMT inside the kernel**: the router computes one row a block with one warp of eight (lever 1a).
- **Bulk**: fewer nodes (fold the grid-8 `qkv_q4k` launches; §9).
- **SIMD/wider loads**: only if §5.4's probe says the request stream, not the bytes, sets the plateau.

## 4. Lever 1 — PDL and launch tails

### 4.1 The toolchain at the pin (`4da6c13`)

- **Device side exists, sm_90 and up.** `cuda_device::grid::dependency::{trigger_dependents, wait}`
  (`cuda-oxide/crates/cuda-device/src/generated/grid_dependency.rs:12-27`), lowered to `griddepcontrol`, with
  `minimum_sm = "sm_90"` (`cuda-oxide/intrinsics/overlay/grid_dependency.toml`; codegen target requirement
  `MinimumSm(90)`, `cuda-oxide-codegen/src/generated_intrinsic_targets/execution_control.rs:374-385`).
- **Raw host launch exists.** `cuda_core::launch_kernel_pdl` sets `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
  through `cuLaunchKernelEx` (`cuda-core/src/api.rs:72-117`); its only caller is `cuda-async/src/launch.rs:281`.
- **The typed launch has no PDL form.** `#[cuda_module]` launchers and `cuda_launch!` take `cluster_dim` and
  `cooperative` (`cuda-macros/src/lib.rs:1053-1060`) and otherwise go through `launch_kernel_on_stream` → `cuLaunchKernel`
  (`cuda-core/src/simt/mod.rs:102-121`). A kernel of ours could take PDL only by hand-building its parameter array.
  Recorded as `docs/upstream/nvlabs-ledger.md` row 42 (feature gap, not a defect).
- **Our PTX cannot carry it.** Every recipe builds `--arch sm_86` (`docs/contrib/dgx-spark.md:157`), and `griddepcontrol`
  needs sm_90. PDL on the Spark rides on the `sm_121` target the maintainers' `archkey`/S2 items add
  (`docs/contrib/dgx-spark.md:321`). The A6000 and 3090 keep sm_86 PTX: no PDL there, Δ = 0 by construction.
- **Graphs keep it.** "Programmatic Dependent Launch can be used in CUDA Graphs via stream capture or directly via edge
  data." ([CUDA programming guide 4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)).
  The same page: "These APIs require at least compute capability 9.0 to provide overlapping execution", and the wait
  "Will block until all primary kernels the secondary kernel is dependent on have completed" (and flushed).

### 4.2 `qwen35moe_router_norm`: the kernel defect (lever 1a)

The entry (`crates/gpu/src/arch/qwen3moe/router.rs:2594-2669`) launches `ROWS = 257` blocks of 256
(`:2494`, `:3414`). In every block: phase A, the FFN norm's sum of squares over `x` on all 256 threads (`:955-978`);
phase B, the normed row into shared memory and one 128-value group quantized in blocks 0–15 (`:980-1036`); then warp 0
alone computes the block's one router row with `f32_lane_partial_1col_w32` (`:1038-1052`; 32 scalar loads a lane in
flight, `crates/gpu/src/q8f32.rs:263-276`, so 4 KB a round and two rounds for k = 2048); then a fence, a ticket and the
last block's single-warp routing (`publish_route_gated`, `:741-806`). Terms `[derived]`:

- **Two waves on 48 SMs.** 47 regs → 5 blocks/SM → 240 slots < 257: 17 blocks run a second wave, each paying the whole
  serial chain (norm ~1 µs, two weight rounds at low parallelism 2–3 µs, fence and ticket ~0.7 µs) ≈ 3.5–5 µs. On the
  A6000 (420 slots) it is one wave. The sibling entry's comment states the same budget for 82 SMs
  (`:2686-2688`, "a grid of 257 blocks stays one wave on 82 SMs").
- **One loading warp of eight.** 240 weight-reading warps hold 240 × 4 KB = 0.98 MB in flight in wave 1: enough (§5.1),
  so wave 1 streams near R_ss: 1.97 MB / 196.8–235.6 GB/s = 8.4–10 µs.
- **The norm in front of the weights.** Every block runs phases A and B before its first weight load: ~1 µs of DRAM idle
  at the start, and 257 × 16 KB = 4.1 MB of L2 reads for `x` and `gain` to produce 2.1 MB of weight reads.
- **The routing tail.** One warp of the last block (`route_warp_gated`, `:565`): eight volatile logit reads, the max
  and exp over 8 values a lane, the f64 sum (eight dependent adds, then a five-step butterfly; f64 runs at a fraction of
  f32 rate on this class of SM), eight divides, then `used` = 8 rounds of a branchy scan over the lane's 8 entries plus
  a five-step (value, id) butterfly, then the f64 slot-order weight sum and nine divides. The 2026-10-08 PTX has 464
  instructions after the ticket's `atom`. As a dependent chain that is ~6,000–10,000 cycles: **T_route ≈ 2.5–4 µs**
  `[derived]`. Its sums are bit-pinned (the module doc's numeric contract), so neither the regrid nor more lanes
  shorten most of it, and PDL cannot hide it (`gate_up` reads the ids it writes).

Model of today `[derived]`: t0 0.95 + wave 1 (ramp and norm ~1–1.5, 1.97 MB at 196.8–235.6 = 8.4–10.0, fence and ticket
~0.5: T_w1 = 9.9–12.0) + T_w2 + T_route = 18.48, so **T_w2 + T_route = 5.5–7.6 µs** and **T_w2 = 1.5–5.1 µs**. ncu's
61.1 % active (alone, base clock, 23.26 µs) is consistent: it leaves 434 SM·µs idle, 31·W + 47·R = 434 with W the time
17 or fewer SMs are busy (wave 2 plus wave 1's drain) and R the one-block tail; R = 2.8–4.6 µs at base clock (the
estimate above × 2.44/2.144) gives W ≈ 7.0–9.7 µs. The tail after wave 1 is two terms of similar size, and the regrid
removes only one of them.

**Design.** Block b, warp w computes row 8b + w with the same `f32_lane_partial_1col_w32` and `gemv_lane_sums`, so
each logit keeps its bits; grid ⌈257 / 8⌉ = 33. Each warp issues its first 32-load chunk into registers before phase
A, so the norm runs under that chunk's DRAM latency. Phase B's quantize ownership (block `t·(k/128) + g`) still has a
block per group: 16 ≤ 33 (the const assert at `:2472` becomes `NORM_K / 128 <= ROWS.div_ceil(8)`). Only the one-token
entry changes; `qwen35moe_router_norm_m` (8 tokens × 16 groups = 128 blocks needed) and `qwen35moe_router_fused` keep
their grids.

**Prediction.** GB10: the regrid removes T_w2, puts the 17 rows' 0.14 MB into the one stream (+0.7 µs at 196.8) and
hides the ~1 µs norm under the first chunk: Δ = −(T_w2 − 0.7 + 1.0) = **−1.8 … −5.4 µs a launch**; T_route stays.
Over 40 launches **−0.07 … −0.22 ms, −0.6 … −1.9 % of 11.50 ms**, center −0.145 ms (−1.3 %) `[derived]`. The center is
outside the ±1 % ruler and the lower edge inside it. §8's eight pairs put the GB10 ruler at ±0.061 ms, under the
band's lower edge, so a null there refutes the T_w2 range: T_w2 under 1.5 µs, T_route over 4 µs. The next action on
a null is the nsys line on the candidate: `qwen35moe_router_norm`'s in-flow µs fell to 13–16.7 → the kernel moved and
the step did not (a term outside §4.2, report it); it did not fall → the second wave was not the tail, and the routing
tail is the term to re-derive from the cc 12.1 SASS. A routing rewrite that keeps the bits (each lane
sorts its 8 entries once, then each round pops the warp argmax of the lanes' heads) removes only the per-round scan,
~0.4–1.0 µs a launch, −0.02 … −0.04 ms: **model says 0, do not open** as its own round. A6000: one wave today, so only the hidden norm (≤ 1 µs) and the L2 traffic cut
(4.1 → 0.54 MB) move: **−1.5 … +0.5 µs a launch, −0.06 … +0.02 ms on a ~5.6 ms step** (193.9 tok/s on the 3090,
`README.md:115`, scaled to 768 GB/s) `[derived]`. On 33 SMs the per-SM read rate is 7.1 GB/s on GB10 (2.9 B/cycle) and
~20 GB/s on the A6000 (11 B/cycle), both far under one SM's L2 port. In flight: one row a warp is 64 four-byte loads a
lane in two rounds of 32 (`f32_lane_partial_1col_w32`, 4 KB a warp a round), 32 KB a block, ~1.06 MB over 33 blocks,
3–9× the 118–353 KB the chip needs at 235.6 GB/s (§5.1) `[derived]`. No kernel on GB10 has been measured at
2.9 B/cycle per SM (the head's 1.7 is the chip rate shared by 48 SMs), so whether 33 SMs reach the chip rate is a
`[derived]` term, and both A/Bs (§7.1 on the A6000, §8 on GB10) settle it; a 33-SM ceiling under the chip rate would
show as a GB10 Δ above the band's −0.07 edge and the nsys µs above 16.7.

### 4.3 The other tails (levers 1c, 1d)

- **`qwen3moe_o_resid_q4k`**, grid `rows / 8` (`crates/gpu/src/arch/qwen3moe/proj.rs:692`), 2048 rows of K = 4096:
  256 blocks on 240 slots, 16 blocks in a second wave. Their 128 warps hold 128 × ≤ 576 B ≈ 74 KB in flight, under the
  118–353 KB the pool needs (§5.1), for two rounds: ≈ 2 µs, against the 1.76 µs tail in §2. A uniform two rows a warp (128 blocks, one wave on
  both cards) saves it: **−0.07 ms, −0.6 %**. Under the ruler: **model says 0, do not open.**
- **`q4k_gemv_sel`** at K = 512 (`n_sb = 2`): the core's guard `sbp < n_sb` (`crates/gpu/src/cores.rs:395-404`) leaves
  lanes with `grp ≥ 2` idle, so a warp reads 288 B with half its lanes, after a dependent `sel` load
  (`crates/gpu/src/q4k_sel.rs:84`). Its 4.60 µs tail is the ceiling of any fix: **≤ −0.09 ms, −0.8 %. Model says 0, do
  not open.** (`q6k_gemv_sel` reads 201 GB/s alone and 176 in flow; that 5.6 µs is a boundary effect, not its body.)

### 4.4 PDL proper (lever 1b)

Per node boundary PDL can hide the next kernel's launch and dispatch, its prologue, and, for a kernel whose first
reads are weights at addresses its predecessor did not write, its first DRAM round trip, under the previous kernel's
tail. The `wait` still waits for completion and flush. That excludes the 40 `gate_up` launches (their rows follow the
ids the router just wrote) but not the downs (ids two nodes back, complete once `gate_up` passed its own `wait`, if
every kernel triggers after its `wait`). Terms `[derived]`: node gap ≤ `c_node` (0.852–0.871 µs on the A6000, rig-log
`docs/bloomery/plan.md` cost-model table); first-byte latency 0.5–1.5 µs (§5.1). Weights-first boundaries 197 ×
0.5–2.0 µs = 0.10–0.39 ms; the other 323 hide only launch and dispatch, 323 × 0.3–0.85 µs = 0.10–0.27 ms. **Total
−0.20 … −0.67 ms (−1.7 … −5.8 %).** It also absorbs 1c and 1d: their second waves become
the next kernel's first blocks.

The band spans 3.5×, so the round does not open on it. The one term: **`c_node` on GB10**, the empty-node price of a
captured graph, `cnode_probe --time` (`crates/gpu-gates/src/bin/cnode_probe.rs`, the `time-gpu-cnode` recipe's binary),
expected 0.6–1.5 µs. Decide: 520 × (`c_node` + 0.5 µs) ≥ 0.35 ms (3 %) → schedule `sparkpdl` (§7.2) after `archkey`;
below → model says 0 for PDL on this model.

## 5. Lever 2 — bytes in flight per SM

### 5.1 The latency

No GPU-side LPDDR5x latency for GB10 is published that I found. Chips and Cheese measured the CPU side
([Inside Nvidia GB10's Memory Subsystem](https://chipsandcheese.com/p/inside-nvidia-gb10s-memory-subsystem)):
"113 ns might feel slow coming from a typical DDR5-equipped desktop, but it's excellent for LPDDR5X", and "Latency from
the CPU side goes beyond 351 ns with the GPU pulling 231 GB/s". From the GPU side the same group reports only an order
([Analyzing Nvidia GB10's GPU](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu)): "Strix Halo's iGPU enjoys better
latency to LPDDR5X than Nvidia's GB10". `[derived]`: the GPU path adds its L1 miss and a 24 MB L2 lookup to the DRAM
time, several hundred ns on NVIDIA parts, so L_idle ≈ 0.5–1.0 µs; under the step's load the queueing the CPU sees
(+240 ns at 231 GB/s) adds to it: **L ≈ 0.75–1.5 µs**. #13's numbers do not fix L: `hostread2`'s `rows_k` holds far more
than it needs (≈ 110 KB/SM), so its rate bounds L only from above.

Little's law, need per SM = 235.6 GB/s × L / 48: **2.5 KB (0.5 µs), 4.9 KB (1.0 µs), 7.4 KB (1.5 µs)**.

### 5.2 Per kernel: in flight today, needed, the gap

In flight per warp is the unique weight bytes one memory round trip carries, read from the code and the PTX order.
The Q4_K walk (`cores::q4k_row_dot_1col`, `cores.rs:341-410`) does one iteration of four super-blocks (576 B a warp);
in the 2026-10-08 PTX of `qwen3moe_o_resid_q4k` each iteration's header loads come first, then the scale decode's
branches, then the `qs` loads and the dp4a, and the second unrolled iteration's loads follow the first one's dp4a — so a
round trip carries 128 B (the four header sectors) or ~448 B (the rest), up to 576 B if ptxas hoists the `qs` loads
over the branches `[unverified: SASS not read]`. 48 registers cannot hold two iterations' 24 weight words plus the
activation words, so `Q4K_ITER_UNROLL = 2` (`cores.rs:324`) does not double what is in flight `[derived]`. The Q6_K walk
(`q6k_sel.rs:104-167`, the same body in `q6k_gemv`) issues its 15 loads together: 420 B a round.

| Kernel | Per warp a round | Rounds a warp | Warps/SM | In flight per SM | Need (0.5 / 1.0 / 1.5 µs) | In flight ÷ need at 1.5 µs | Measured steady |
|---|---|---|---|---|---|---|---|
| `gate_up_swiglu_q4k` | 128–576 B | 4–8 (gate then up, K 2048) | 40 | 5.1–23.0 KB | 2.5 / 4.9 / 7.4 KB | 0.7–3.1 | 184 in flow |
| `qkv_q4k` | 128–576 B | 2–4 | 32 | 4.1–18.4 KB | " | 0.6–2.5 | 196.8 (slope) |
| `o_resid_q4k` | 128–576 B | 4–8 (K 4096) | 40 | 5.1–23.0 KB | " | 0.7–3.1 | 177 in flow, wave tail |
| `q4k_gemv_sel` | 64–288 B (half the lanes) | 1–2, after a dependent `sel` load | 40 | 2.6–11.5 KB | " | 0.3–1.6 | 163 |
| `q6k_gemv_sel` | 420 B | 1, after `sel` | 48 | 20.2 KB | " | 2.7 | 201 alone |
| `q6k_gemv` | 420 B | 4 | 48 | 20.2 KB | " | 2.7 | 186 |
| `head_q6k_argmax` | 420 B | 4 | 48 | 20.2 KB | " | 2.7 | **199.2**, 99.8 % active |
| `router_norm` | 4 KB (warp 0 only) | 2 | 5 | 20 KB during the gemv | " | 2.7 | 117 (chain, §4.2) |
| `hostread2` `rows_k` (#13) | ≈ 2.3 KB (two rows) | 1 | 48 | ≈ 110 KB | " | 15 | 235.6 |

### 5.3 What the table says

Little alone is ambiguous for the Q4_K kernels (0.6–3.1× the need, by which phase is outstanding). The measured rates
decide it: across in flight from ~4 KB (qkv, header phase) to 20 KB (the Q6_K head, one phase) the steady rate stays at
**189–199 GB/s**, and a kernel that holds 2× more does not run 2× faster — the head with 20 KB at 99.8 % active runs at
199.2, the qkv slope with less gives 196.8, and `q6k_gemv_sel`, one round of 20 KB, reads 201 alone. A rate that is
flat in in-flight bytes is past the knee: raising bytes in flight per SM to cover latency (more loads per thread before
use, more blocks per SM, wider grids) **predicts 0 on the layer gemvs and the head. In this Little's-law form, model
says 0, do not open.** The one kernel under the knee, `q4k_gemv_sel` (163), is worth ≤ 0.09 ms (§4.3), under the ruler.

The same code shape can still pay through another term: a Q4_K row's header and `qs` loads issued in one phase, not
two, change the order the requests reach the LPDDR5x rows, not how many are outstanding. That is the term of §5.4,
undecided until the probe's arm P; it is not a Little's-law lever and this table does not predict it.

### 5.4 The term the 1.63 ms is in

Every big gemv sits at 196.8–199.2 GB/s, 84 % of the plain read, with no SM unit near its demand (§2) and in flight past
the knee (§5.3). What stays different between them and `rows_k` (235.6) is the request stream the memory system sees:

- **(a) Access width.** Ours are 4-byte loads (`ld.global.b32`, no vector load); `rows_k` uses 16-byte `__ldg`, one
  instruction requesting 512 contiguous bytes. L2 traffic is equal (1.01–1.05 × the fills), so a cost here would be in
  how the L2 forms DRAM requests (a Q4_K header sector requested one phase before the rest of its line).
- **(b) Row pieces.** A warp fetches its row in 2–8 pieces separated by a round trip (Q6_K head: four 420 B pieces of a
  1680 B row; o_resid up to eight), while `rows_k` requests each whole row at once. Between pieces the DRAM page of the
  row may close; LPDDR5x pays an activate per reopened page. On the A6000 a Q6_K head, the same walk, reads 701 GB/s
  (91 % of 768; the large launches 567–701, rig-log `docs/bloomery/plan.md` cost table) against 199.2 on GB10 (73 % of
  273, 85 % of the plain read), so the walk loses more on LPDDR5x than on GDDR6.

Neither is fixable by derivation. **The one term: the read rate of `hostread2`'s `rows_k` with our request stream**, on
the Spark, same buffers: arm W (4-byte loads in the Q4_K lane-to-word map, whole row at once), arm P (16-byte loads,
the row in four pieces chained through a dependent register), against today's arm (16-byte, whole row, 235.6).
Expected `[derived]`: if (a), W ≈ 190–205 and P ≈ 230–236; if (b), P ≈ 190–205 and W ≈ 230–236; if neither, both
≥ 228 and the 1.63 ms goes to a reading round with ncu memory-side metrics before any kernel work. Next action: (a) →
round `lpddrwidth`: `U32x4` loads for the Q4_K walk (`cuda_device::vector::U32x4` is in use, `crates/gpu/src/iq.rs:85`;
the header and `qs` offsets are 16-byte aligned since 144 = 9 × 16), Q6_K needs a load-time layout (210 B super-blocks
are 2-byte aligned); (b) → round `lpddrrows`: issue a row's pieces together (registers allow it at Q6_K; the Q4_K walk
needs the decode moved after the loads). Either predicts up to −1.4 ms on GB10 (1978 MB from 196.8 to ≥ 230 GB/s,
−12 %) and 0 ± ruler on the A6000 (the q3gemma precedent: −39 % L1TEX wavefronts at ~700 GB/s moved no cycles).

## 6. Reference designs

- **ik_llama.cpp** (`~/repo/ik_llama.cpp`, the numeric oracle): `mul_mat_vec_q` at one column runs one row a block
  (`rows_per_cuda_block = 1` for `ncols_y < 4`, `ggml/src/ggml-cuda/mmvq-templates.cuh:84-86`), four warps on NVIDIA
  (`:444-447`), 8 super-blocks a loop pass (`blocks_per_iter = vdr * nwarps * WARP_SIZE / qi`, `:93`, `vdr = 2`,
  `vecdotq.cuh:393`), so a K = 2048 Q4_K row is one pass; each thread's `qs` loads are 4-byte `int`s (`vecdotq.cuh:748-750`)
  issued before the scale decode. No PDL, no integrated-memory path in its mmvq.
- **mistral.rs** (`~/repo/mistral.rs`): `mistralrs-quant/kernels/mmvq_gguf/mmvq_gguf.cu` at one column, 8 warps × 2 rows
  a block (`MMVQ_NWARPS_SINGLE_COL_PLAIN 8`, `MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN 2`, `:15-16`; fused GLU 4 × 2,
  `:17-18`), the same ggml `vec_dot` lineage and 4-byte loads. PDL appears only in `blockwise_fp8_cutlass_sm90.cu`, not
  in a gemv. No LPDDR or integrated path.
- **exllamav3**: not on this Mac (`/home/user/exllamav3-src` is on the box). Not read; no claim about its gemv.

Where the three diverge: ik reduces one row across four warps through shared memory, mistral.rs two rows across eight,
ours one row a warp with no shared memory but a per-warp serial walk. None keeps more bytes in flight per SM than ours
by a large factor, none uses wider loads in the Q4_K gemv, none has a unified-memory design. The design followed here:
ours for the bits (warp-per-row, pinned), ik's "a row's loads before its decode" for §5.4 (b) if the probe picks it.

## 7. Round specs

### 7.1 `sparkrouter` (opens now; class: occupancy/geometry — grid and per-block work change, each row's arithmetic does not)

- **File.** `crates/gpu/src/arch/qwen3moe/router.rs` only: the `gated` one-token entry `qwen35moe_router_norm`
  (`:2594`), `norm_row_body` (`:926`, the row → warp map and the hoisted first chunk, or a sibling body for the
  8-rows form if `_m` must keep the old one), the const assert at `:2472`, the launcher's grid (`:3369-3420`). Not `_m`,
  not `_fused`, not the qwen3moe (128-row) entries.
- **Owning gates.** `just gate-gpu-qwen35moe-moe` (the 257-row gated router against the host rule: logits, ids,
  weights), `just gate-gpu-qwen35moe-e2e` (decode node count 520 unchanged, eager = replay, logits bit-identical),
  `just gate-ptx-spill` (the entry's row stays 0/0). `just ptx-scan generate_qwen3moe --features gpu` shows the one entry
  moved; blocks/SM may fall to 1–4 with the hoisted registers, which 33 blocks on ≥ 33 SMs tolerates — write the scan's
  regs/blk next to that.
- **FAIL-first.** (1) map warp w to row 8b + (w + 1) % 8 → `gate_qwen35moe_moe`'s logits red; (2) feed the hoisted chunk
  as zeros → red; (3) compare the ticket against `ROWS` instead of the grid → no block routes, the ids go stale, red
  (pins that the routing follows the new grid). Each reverted, green.
- **A/B on the A6000** (non-inferiority, the A6000 Δ sits inside the ruler): card `docs/cards/sparkrouter-a6000.card`,
  `kind: noninf`, `term: decode tok/s @ n=64, depth 6, A6000`, `predict: -0.4..1.0 %` (tok/s; the step's
  −1.0 … +0.4 %), `rounds: 6`, `margin: 1.0 %`, then
  `BLOOMERY_AB_ROUNDS=6 BLOOMERY_BOX_ENV='BLOOMERY_AB_LOAD=arm BLOOMERY_LEASE_CARD=docs/cards/sparkrouter-a6000.card' just depth-gpu-qwen35moe 6 bin:/root/repo/bloomery-sparkrouter-base/target/release/generate_qwen3moe:6`
  (`just ab-decode` times the CPU binary `bloomery-decode`, `tools/ref/lease.sh:859`, so the GPU path's same-lease
  A/B is the depth runner's `bin:` arm; `BLOOMERY_AB_LOAD=arm` gives each arm its own process). The runner's ratio is
  ours (the candidate tree) / `bin:<base>`, so + means the candidate is faster; the card's band is in that direction.
- **On GB10.** The contributor's command, §8, predicted band −0.07 … −0.22 ms.

### 7.2 `sparkpdl` (after `archkey` and the GB10 `c_node` reading; class: launch/dependency — no arithmetic moves)

- **Prerequisites.** (i) `--arch sm_121` builds of the qwen3moe bins (`archkey`, maintainers). (ii) A fork commit on
  `bloomery`: a PDL form of the typed launch (`#[cuda_module]` / `cuda_launch!` option next to `cooperative`, through
  `launch_kernel_pdl`), with FAIL-first: a captured two-kernel graph whose edge reads back as programmatic, and red with
  the option off. (iii) Ledger row 42 moves to that commit.
- **Files.** The Qwen3.6 decode chain's launchers (`crates/gpu/src/arch/qwen3moe/{proj,experts,router,body35}.rs`,
  `crates/gpu/src/{q4k_sel,q6k_sel}.rs`, the flash and delta launchers), and in each entry: `wait()` before the first
  read of a predecessor's output, `trigger_dependents()` after the block's weight loads are issued, the first weight
  chunk issued before `wait()` where the registers allow. All of it under the sm_121 target; the sm_86 PTX unchanged
  (`just ptx-scan` identical on both box bins — the A6000/3090 proof).
- **Gates.** `just gate-gpu-qwen35moe-e2e` (bits, node count 520, eager = replay). FAIL-first: drop one `wait()` and
  add a spin in the producer's last block → the consumer reads the previous step's activation → red.
- **A/B.** GB10 only (§8); predicted band from §4.4 narrowed by the measured `c_node`.

### 7.3 `lpddrwidth` / `lpddrrows` (only on §5.4's probe outcome)

Files `crates/gpu/src/cores.rs` (Q4_K walk), `q6k_sel.rs`, `lib.rs` `q6k_gemv`, the head; gates the qwen35moe and
qwen3moe family gates bit-identical (only load width or order changes), `gate-gpu-e2e`; A/B the A6000 noninf as 7.1
and GB10 by §8. Predicted up to −1.4 ms on GB10.

## 8. The contributor's command

Two trees built from the same commit and the candidate (`~/base`, `~/cand`), `$M` the model file; eight interleaved
pairs, page cache dropped before each process, graph mode, the comment-1 flags:

    for r in 1 2 3 4 5 6 7 8; do for t in base cand; do sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; BLOOMERY_REF_MODEL=$M ~/$t/target/release/generate_qwen3moe --prompt "The capital of France is" -n 64 --prefill pass --mode graph --time 2>&1 | grep '^SMOKE' | grep -o 'p50_ms=[0-9.]*' | sed "s/^/$t r$r /"; done; done

Rule: Δ = mean(cand p50) − mean(base p50); ruler r = t(0.975, 14) × max(0.057, s) × √(2/8) = 2.145 × max(0.057, s) × 0.5,
with s the pooled SD of this run's two arms and 0.057 ms the SD of comment 2's three processes (a weak estimate, so a
noisier session widens r and cannot make a false "real"). At s ≤ 0.057, r = ±0.061 ms, under the band's −0.07 edge.
**Δ ≤ −r**: the lever is real on GB10; inside the band (−0.07 … −0.22 for `sparkrouter`) the model holds, outside it
the report names which §4.2 term was off. **|Δ| < r**: the band is refuted; the next action is §4.2's (the nsys line on
the candidate). **Δ ≥ +r**: reject.

## 9. Open terms, in the order they decide something

1. §5.4's probe on the Spark (rows_k W and P arms): decides whether 1.4 ms is reachable and by which round.
2. GB10 `c_node` (`cnode_probe --time` on the Spark): decides `sparkpdl`.
3. The cc 12.1 JIT's registers for the seven entries (`cuFuncGetAttribute(NUM_REGS)` on the Spark): the waves column
   assumes the sm_86 scan's blocks/SM.

## 10. Measured after the design (contributor @juliankang4, #17 comment 6105096461)

Interleaved runs of the contributor's own script on one Spark, 4 rounds, ids identical across arms, SM clock capped at
1989 MHz; leads, not lease-runner numbers.
- §5.4's row-locality form, as a per-warp `prefetch.global.L2` of every 128-byte line of the weight row before the
  walk, on every decode gemv: step 11.97 → 10.35 ms (−1.62 ms). That is past this design's −1.4 ms upper bound, so the
  term was row locality and the bound was low; open term 1 is decided. Lookahead into the next wave (+2.4 %, +6 %), a
  side-stream L2 fill (+7.7 %), a prefetch node before the short kernels (±0.1 %) and `launch_bounds(256, 6)` (spills,
  +27 %) lost. The A6000 Δ is not measured; the landing needs the noninf A/B of §7.1's form.
- PDL on the same gemvs on top of the prefetch: +0.9 … +1.1 % (slower). `sparkpdl` stays closed after the prefetch.
- The router's top-k tail (each lane keeps its best untaken expert): Qwen3.6 +0.17 % ± 0.19 % (model said 0, §4.2);
  Qwen3.8 (512 experts) −0.89 % ± 0.17 %.
