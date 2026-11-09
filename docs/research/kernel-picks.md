# Kernel picks from facts: a derive-first launch picker (answer to #1151)

Status (2026-11-04): a proposal beside #1151's load-time race, not built. Its band (±10 %) is derived, not measured. The first step is the calibration both designs need: force every candidate on the real shapes on sm_86 and sm_121a, and set the model's ranking against the measured one. That result picks between this design and the race.

The race is built in `crates/gpu/src/autotune.rs`, with the MXFP8 gemv of four to eight columns as its first site; `bench_picks` forces each way it races there (`enqueue_e4m3_gemv_way`).

Status: proposal, 2026-11-04, written on main `6a3570613`. Nothing here is built. Tags: `[measured: source]` is a
number someone timed, with its source; `[derived]` is arithmetic on the tree's facts; `[datasheet]` is a vendor figure
that the load would read from the driver instead. Register, shared-memory and blocks-per-SM figures are from the
`ptx-scan --no-jit` of `bloomery-serve` on 2026-11-03 (ptxas 13.3.73, `arch=sm_86`). The JIT's own counts can differ
from these (`docs/research/sparkdec.md:62`).

## 0. Result

Most of #1151's choices are arithmetic on facts known before any kernel runs. The facts are the shape (from the file
and the plan), each entry's occupancy (from the build, and from the driver's occupancy query on the loaded function),
and the card (driver attributes). The rule computes each candidate's terms: DRAM bytes, L2 requests, launches, and
waves = ⌈blocks / (SMs × blocks per SM)⌉. It picks a candidate only where the gap between candidates is wider than the
model's band. Inside the band, today's rule stands (the *incumbent*). It moves only through a per-arch row of measured
constants that ships in the binary. That row comes from a forced-candidate kernel bench on that arch's card, run once.
The load times nothing and writes no cache. Two loads with the same build, driver and card make the same picks. Every
load prints its picks and a digest of them. `fixed` (off) runs today's rules verbatim.

Residue (what a model cannot settle, §4): R1 the MXFP8 plain-against-staged walk at 2 to 4 columns; R2 the Q8_0 slab
entry near one wave and one or two tiles an expert; R3 the per-arch constants themselves (c_node, the walks' read
shares, the L2 request rate); R4 routing skew on routed GEMMs (the pick is fixed on the expectation, and one histogram
per model checks it).

## 1. What decides today

| Site | Rule today | File | Tuned on |
|---|---|---|---|
| MXFP8 gemv, m columns | m = 1 `mxfp8_gemv`; m = 4 with k % 512 = 0 and x 16-byte aligned `mxfp8_gemv_mcol_staged`; m = 5..8, same conditions, `..._staged_wide`; else `mxfp8_gemv_mcol` | `crates/gpu/src/q8f32.rs:3437-3463`, `STAGED_COLS` `:1550-1557` | GB10 only. `ea840911c`: staged +4 to 7 % at m = 4, level at 3, −5 to −10 % at 2 `[measured: juliankang4, GB10]`. `4928e3a0e` widened the stage to 5..8 |
| Q8_0-plane GEMM | `gemm_q8_0p_slab` when grid ≥ one wave (SMs × the entry's blocks per SM, read from the driver at load) and n_slots > 64 × n_experts; else `gemm_q8_0p` | `crates/gpu/src/gemm/gemm32.rs:1640-1658`, wave `crates/gpu/src/gemm/kernels32.rs:2163-2182` | GB10 only. `e78ce30ce`: plain bound by L2 requests (L2 tag stage 86 % of peak, IMMA 21 %); slab loses up to 8 % at one tile an expert or under one wave `[measured: juliankang4, GB10]` |
| Slab raster | groups of `GEMM32_GROUP` = 16 tiles for an expert of ≥ `GEMM32_RASTER_BYTES` = 8 MiB of planes | `crates/gpu/src/gemm/gemm32.rs:137-144` | sized for GB10: "a quarter of a 24 MiB L2" (`:137-139`) |
| F32 / BF16 gemv arm | F32 m = 1 `f32_gemv`, m > 1 the F32 tile (token-major); BF16 `bf16_gemv_mcol` | `crates/gpu/src/site.rs:483-497` | none recorded |
| F32 / BF16 wide arm | the exact tile (`f32_tile_gemm`, `bf16_tile_gemm`); on cc 12 BF16 and E4M3 take the MMA tile | `crates/gpu/src/gemm/f32tile.rs:1-60`, `crates/gpu/src/lib.rs:2260-2267` | the MMA pick is an arch rule outside the bit-equal set |

Two facts follow. First, two of today's rules already decide from facts and not from a timing at load: the slab wave
comes from the driver, and the staged walk's eligibility from k and an address. Second, both crossovers were measured
on one arch. sm_86 (A6000, 3090) and sm_120 (5090) run GB10's thresholds unmeasured. #1151 is right that this is a gap.
The question is only how to close it.

The tree already has an owner of this kind: #1082's `runtime::costs`. It computes the card's side "from the device" (a
pure function of driver attributes, so nothing is stored) and keeps a measured value under the device's key so later
loads price with the same bits (`crates/runtime/src/costs.rs:1-21`). Its attribute reader is
`crates/gpu/src/host/costs.rs:28-63` (SMs, clock, cc, memory clock, bus width) and its key `:86-109`.

## 2. The facts, and when each is known

| Fact | Source | Known | Same on every load of the key |
|---|---|---|---|
| rows, k of a site | the file header (`chain::site::file_site`, read before any upload) | plan | yes |
| m: the widths a site launches at (decode 1, verify widths, two sequences 5..8, the ubatch) | the plan; a finite set | plan | yes |
| n_experts, n_slots of a route | the plan (n_slots = tokens × top_k) | plan | yes |
| slots per expert | the route table on the device | run time, data | no: priced by expectation (R4) |
| regs, smem, threads, spill of an entry | `ptx-scan` (ptxas `-v`), `tools/ref/ptx-shapes.tsv` and per-arch tables | build | yes |
| blocks per SM of the loaded entry | the driver's occupancy query on the JIT'd function, as `q8_0p_wave` does (`kernels32.rs:2167-2182`) | load, µs | yes for one (build, driver) |
| SMs, SM clock, cc, DRAM rate | driver attributes (`host/costs.rs:28-63`) | load | yes |
| L2 bytes | `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`, read by the load-time race (`crates/gpu/src/autotune.rs`) to size its rotation | load | yes |
| buffer alignment | fixed by the load's allocations, before capture | load | yes |
| c_node, the walks' read shares, the L2 request rate, a walk's per-step latency | measured once per arch (§4, R3) | release | yes (shipped) |

Every fact the picks need is known before capture. Only the slots per expert are data, and the pick cannot depend on
them anyway: it is fixed before capture (#1151 §4.3).

## 3. The model

For a candidate c at shape (rows, k, m) on a card:

    t_c = max( B_c / (R_dram · s_c),  Q_c / R_l2,  I_c / R_issue,  W_c · L_c ) + N_c · c_node
    W_c = ⌈ blocks_c / (SMs · b_c) ⌉

- B_c: bytes from DRAM. R_dram: the card's DRAM rate (memory clock × bus). s_c: the walk's read share.
- Q_c: L2 requests. R_l2: the card's L2 request rate.
- I_c: issued instructions (PTX counts, per step × steps).
- W_c: waves. b_c: blocks per SM of the loaded entry. L_c: one block's dependent chain (the latency floor).
- N_c: launches. c_node: one graph node, 0.852 µs on the A6000 `[measured: rig-log plan.md:71, B9]`.

The pick is the argmin over the eligible candidates, with three rules on top:

1. **Eligibility is exact.** k % 512, a 16-byte base, the cc, the alignment of a plane. An ineligible candidate is
   never priced.
2. **Dominance is exact.** If one candidate's every term is ≤ another's, it wins with no constant.
3. **A gap inside the band keeps the incumbent.** The band is the error of the shared constants: one read share for
   all kernels spans 0.69 to 0.85 (`crates/runtime/src/costs.rs:44-48`), so ±10 % `[derived]`. With a walk's own
   measured row it narrows to that bench's spread.

The busiest unit is not the bound until the step's cycles are its demand. If every candidate's largest demand term is
well under its measured time, the kernel is latency-bound. Then the demand terms cannot rank the candidates, and the
pair belongs to the residue (AGENTS.md, rig-log 09-26#q3gemma-ab: L1TEX wavefronts −39 %, cycles unmoved).

### 3.1 MXFP8 gemv: plain, staged (stage of 4), staged-wide (stage of 8)

All three: one launch, ⌈rows / 8⌉ blocks of 256 threads, one warp a row, four steps of code words hoisted
(`Q8_STEP_UNROLL`, `q8f32.rs:928`), the same weight bytes rows × k × 33/32.

| Term | plain `mxfp8_gemv_mcol` | staged | staged-wide |
|---|---|---|---|
| regs / smem / blocks per SM, sm_86 | 72 / 0 / 3 | 48 / 4096 / 5 | 47 / 8192 / 5 |
| DRAM bytes | equal | equal | equal |
| launches | 1 | 1 | 1 |
| x read on chip, bytes a weight byte | 3.9 m via L1, every warp | 3.9 m from shared + 0.49 m via L1 | same as staged |
| barriers | none | one a step | one a step |
| eligible m | 2..8 | 1..4 | 1..8 |

The x terms are from the code (`q8f32.rs:1593-1700`). A warp-step reads 132 weight bytes (128 codes, 4 scales) and
512 m bytes of x. The staged walk loads the x once a block of eight warps.

Waves and DRAM floors at the three Qwen3.8 verify shapes (`gate_mxfp8.rs:1123`). Blocks per SM are sm_86's. GB10's
waves use sm_86's blocks per SM, which the sm_121 scan must confirm. Floors use read share 0.85 at the datasheet rates
(A6000 768 GB/s, `docs/v41-placement.md:10`; GB10 273 GB/s, `docs/research/tierengine.md:236`). SM counts: 84 and 48
(`docs/research/sparkdec.md:61-63`). All `[derived]`.

| rows × k | MB | blocks | A6000 waves plain / staged | A6000 floor | GB10 waves plain / staged | GB10 floor |
|---|---:|---:|---|---:|---|---:|
| 2560 × 6144 | 16.2 | 320 | 1.27 → 2 / 0.76 → 1 | 24.8 µs | 2.22 → 3 / 1.33 → 2 | 69.9 µs |
| 10240 × 2560 | 27.0 | 1280 | 5.08 → 6 / 3.05 → 4 | 41.4 µs | 8.89 → 9 / 5.33 → 6 | 116.5 µs |
| 12288 × 2560 | 32.4 | 1536 | 6.10 → 7 / 3.66 → 4 | 49.7 µs | 10.67 → 11 / 6.40 → 7 | 139.8 µs |

What decides:

- **Exact:** eligibility. Also staged against staged-wide at m ≤ 4: equal reads, equal blocks per SM on sm_86, twice
  the shared memory and twice the predicated FMA lines (64 against 128 `fma` in the scan). The narrow stage dominates.
  Today's rule agrees.
- **Band:** plain against staged at m = 2..4, and plain against staged-wide at 5..8. DRAM bytes and launches are
  equal. The candidates differ in the wave tail, in L1 tag lookups against shared loads, and in the barrier a step.
  GB10's measurement puts the whole effect inside ±10 % and crossing zero between m = 2 and m = 4. That is R1.
- **A term GB10 did not stress:** at the DRAM floor, the A6000 feeds each SM 7.8 GB/s against GB10's 4.8 `[derived]`.
  So the on-chip x demand per SM at m = 8 is about 134 B a clock at 1.8 GHz. That is near the 128 B a clock of the SM's
  L1/shared array `[datasheet]`. Both walks read x from that one array, so neither reaches the floor there, and the
  remaining terms are again inside the band. The crossover m* on sm_86 cannot be copied from GB10. It must be measured.

### 3.2 Q8_0-plane GEMM: `gemm_q8_0p` against `gemm_q8_0p_slab`

| Term | plain | slab |
|---|---|---|
| regs / smem / blocks per SM, sm_86 | 106 / 33312 / 2 | 114 / 33312 / 2 |
| one wave, A6000 | 168 blocks | 168 blocks |
| L2 requests a row-step (64 values) a tile | 3 (two 16-byte code chunks, one 2-byte scale) | 1.25 (one 64-byte read, an 8-scale window a 4 steps) |
| DRAM fetches of a slab | from the block order, the wave and the L2 size | the same, with the raster's groups of 16 tiles |
| instructions | base | "a few" more a step (`e78ce30ce`) |

What decides:

- **Equal:** occupancy (2 and 2 on sm_86), launches (1), MACs, bits (the order is a function of K alone,
  `gemm32.rs:52-55`).
- **Computable, no constant:** how many times each weight slab is fetched from DRAM. `raster32` is a pure function
  (`gemm32.rs:154-170`). A small simulation of the grid order, with the resident wave and an L2 of the card's size,
  counts the fetches of each candidate. Where the stack is larger than L2 and many tiles share each slab, the fetch
  ratio is a large integer factor, and the pick is exact.
- **Needs one constant:** the L2 request term. Plain sends 2.4 times the requests of the slab entry. That matters only
  when the requests are the bound, which needs the card's L2 request rate (R3). Today's proxy "grid ≥ one wave and
  more than one tile an expert" stands for "the requests are the bound". The model replaces the proxy with the demand
  itself.
- **Band:** a grid near one wave, and one or two tiles an expert. GB10 measured ≤ 8 % there. That is R2.
- **A finding outside the race's reach:** the raster's constants encode GB10's L2. On a 6 MB L2 (A6000, 3090
  `[datasheet]`), a group's activation columns at K 6144 (6.3 MB, `gemm32.rs:137-139`) are larger than the whole L2. A
  stack between 6 and 8 MiB is assumed to stay in L2 across tiles and cannot `[derived]`. Both constants should be
  functions of the L2 attribute. This changes the block order and not the bits, so it is not a candidate to race. The
  model sets it from one attribute.

### 3.3 F32 / BF16: exact tile against repeated gemv

| Term (sm_86) | `f32_gemv` / `bf16_gemv` | `f32_tile_gemm` / `bf16_tile_gemm` |
|---|---|---|
| regs / smem / blocks per SM | 64 / 0 / 4 and 60 / 0 / 4 | 214 / 49152 / 1 and 202 / 49152 / 1 |
| columns a launch | ≤ 8 (`f32_gemv` row-major; `bf16_gemv_mcol` either layout) | 32 a column tile (`f32tile.rs:86-87`), any m |
| weight reads | once a launch | once a column tile |

What decides:

- **Narrow arm (m ≤ 8), F32:** not a speed pair. `f32_gemv` writes row-major and the site needs token-major
  (`site.rs:483-488`), so the two are different products unless a copy is counted.
- **Narrow arm, BF16:** equal bytes, one launch each. The grids differ: 320 blocks at 4 an SM against 80 at 1 an SM
  for 2560 rows on the A6000 `[derived]`. Inside the band, so the incumbent (`bf16_gemv_mcol`) stays.
- **Wide arm (m > 8):** exact. Repeated gemv takes ⌈m/8⌉ launches and reads the weight ⌈m/8⌉ times (from DRAM when it
  is larger than L2). The tile takes 1 launch and one read a column tile from L2. Example: a 256 × 7168 F32 weight
  (7.3 MB, over a 6 MB L2) at m = 512. Repeated gemv: 64 launches, 470 MB from DRAM, about 0.72 ms + 0.05 ms of nodes
  on the A6000. The tile's floor is its FMA term, 48.6 µs at the card's FP32 peak. The tile loses only if it runs below
  6 % of that peak `[derived]`. Today's rule already takes the tile, and the model confirms it.
- **Not candidates:** the F32 and BF16 store twins (`gemm_q8_0p` and `gemm_q8_0p_bf16`, `7a709c9ed`). The consumer's
  form picks them, and their bits differ. The MMA tile on cc 12 (`lib.rs:2265`) agrees with the exact tile only within
  a band. #1151 already keeps both out.

### 3.4 Summary

| Pair | Facts that decide | All known before capture | Exact | Band |
|---|---|---|---|---|
| MXFP8 staged vs staged-wide, m ≤ 4 | smem, FMA lines, blocks per SM | yes | yes (dominance) | — |
| MXFP8 plain vs staged, m 2..4; vs wide, 5..8 | waves, L1 tag vs shared, barrier a step | yes | eligibility only | ±10 % (GB10's measured span); R1 |
| Q8_0 slab vs plain, stack > L2, many tiles a slab | DRAM fetches by raster simulation, L2 bytes | yes | yes (integer fetch ratio) | — |
| Q8_0 slab vs plain, near one wave, 1–2 tiles an expert | L2 requests vs the card's request rate | yes, plus one constant | no | ≤ 8 % (GB10); R2 |
| Slab raster group and threshold | L2 bytes | yes | yes (no bits move) | its effect needs one A/B |
| F32/BF16 tile vs gemv, m > 8 | bytes ⌈m/8⌉ : 1, launches ⌈m/8⌉ : 1 | yes | yes | — |
| BF16 tile vs gemv, m ≤ 8 | grid fill | yes | no | inside the band; incumbent |

## 4. The residue and the smallest measurement

Weight values do not set the speed of any candidate here. The decode is branch-free, and the exceptional-operand
replay belongs only to the MMA tiles, which are outside the set. So every bench runs on synthetic data of the real
shape. The one data-dependent input is routing (R4).

- **R1. MXFP8 crossover m\* per arch.** Bench: each walk forced, at the three verify shapes and `attn_q_b`
  (k = 1536), m = 1..8, with an A/A arm in the same sitting for the bench's own spread. Output: one row per arch,
  m*(k, waves). sm_86: us, on the A6000 under the timing lease. sm_121: our Spark, or juliankang4's GB10 numbers
  re-run through the forced bench. sm_120: a contributor's 5090 as a `gpu-evidence` paste.
- **R2. Slab crossover.** Bench: both entries forced, over a sweep of grid / wave (0.5 to 4) and tiles an expert (1,
  2, 4, 16), at K 2560 and 6144. `gate_gemm`'s wave arms (`gate_gemm.rs:3742-3758`) are already placed at wave − 1,
  wave and wave + 1. Today each runs only the side the rule picks. Same owners as R1.
- **R3. The constants.** c_node is measured on the A6000 only. Each walk's read share (0.69 to 0.85 across kernels
  today). The L2 request rate (the slab pair's bound). A walk's per-step latency floor L (the latency term). All are
  fitted from the R1/R2 sweeps, so no extra sitting. Each row is keyed by (cc, entry, the entry's PTX md5) and carries
  its provenance (card, driver, commit, command, date) as a `PIN` line does.
- **R4. Routing skew.** A routed launch's tiles an expert are counted on the device. The pick is fixed before capture
  on the expectation: Σ over experts of E[⌈n_e / 64⌉], with n_e ~ Binomial(n_slots, 1/E), the same form as the
  host-union model's (rig-log plan.md:77). A real prompt's route histogram, dumped once per model, checks it. A
  load-time race cannot see the serving distribution either. It would time whatever routing its probe used.

## 5. Where it lives, how it prints, how constants ship, what off means

- **Owner.** A pure core, `crates/placement/src/kernels/pick.rs` (proposed), beside the type-to-entry table. It takes
  (site shape, m class, entry facts, card facts, arch row) and returns (candidate, source, the terms). It has no card
  and is tested on the Mac. Its device half, `crates/gpu/src/pick.rs` (proposed), beside `site.rs`, fills the facts
  once a load. The card facts come through `host::costs::card_rates` extended by L2 bytes. The entry facts come from
  the driver's occupancy query on each candidate's loaded function, the call `q8_0p_wave` makes today. Models give
  shapes only. The launchers take the pick as an argument: the `match` at `q8f32.rs:3443-3463` and the
  `q8_0p_entry` choice at `gemm32.rs:1646-1658` become the core's output, and a launcher refuses an ineligible pick by
  name. The picks for every (site, m class) the plan can launch are computed at load into a table the launch reads by
  index. The step does no load-time work.
- **Print and records.** One `kernel pick` line a distinct (family, rows, k, m) at load: picked, incumbent, source
  (`exact`, `table <cc> <md5>`, `incumbent no-row|stale|band`, `fixed`), the predicted gap. A `pick_digest` goes in
  the load summary and the server's status JSON. A new record kind in `crates/gpu-gates/src/record.rs`, read through
  `tools/bloomery/records.py`. The timing runners print each arm's digest. A comparison across two digests says so.
- **The constants table** is compiled into the binary (`include_str!` of one TSV, every arch's rows), so each release
  archive (sm_86 portable, GB10 native) carries all of them and uses the row of the card it runs on. A row whose md5
  no longer matches the entry's PTX is stale: the pick falls back to the incumbent, and the record says `stale`. A
  check in the release batch, which already scans, holds the table's md5s to the scan's md5 block. A self-tested
  Python tool, listed in `check-recipes`.
- **Off.** A parsed lever row in `crates/levers/src/registry.rs`, `BLOOMERY_KERNEL_PICK=rule|fixed`, default `rule`.
  `fixed` runs today's rules verbatim (the core keeps them as its fallback, so `fixed` is no extra path). Neither mode
  times anything or reads or writes a file. Under `rule`, the picks equal `fixed`'s until a measured row says
  otherwise (§6, step 2).

## 6. How it is proven

| Step | What | Proof | AGENTS.md class |
|---|---|---|---|
| 0 | Forced-candidate bench arms. `gate_mxfp8 --bench-kernels` exists (`gate_mxfp8.rs:99-108`), but it calls `enqueue_e4m3_gemv`, the rule (`:1165`), at m = 1..4 only (`:1124`). It prices the rule's pick, not each walk. `gate_gemm`'s q8_0p arms do the same. Add `--bench-force <entry>` through the launcher's pick argument. This is also #1151 §5's first step. | `just ptx-scan --no-jit` equal to base; the bench asserts nothing | move/split (host only) |
| 1 | The pure core with the incumbent and the model; no engine caller | Mac tests: a golden pick table over a shape × arch grid; "picks = fixed on every shape the gates and benches run today" (verify shapes, `attn_q_b` m 1..6, `gate_gemm`'s g32 and wave shapes) for every arch with no measured-win row. FAIL-first: `STAGED_COLS` 4 → 3 turns it red | add |
| 2 | Wire it: the engine asks the core; the table ships empty of wins | `just canon-check BASE` (every entry the same program); graph node count, eager = replay and the e2e set identical; `gate-gpu-mxfp8`, `gate-gpu-gemm` bit-identical | move (host dispatch) |
| 3 | Each changed pick, one at a time: a measured row lands | forced bench on the real shapes on that arch's card, under the lease with a prediction card, the gap outside the bench's A/A spread; then a same-lease `just ab-decode` with `BLOOMERY_AB_ENVS="BLOOMERY_KERNEL_PICK=fixed"` on the model that runs the shape, C = 1 and C = 2 as #1151 §5 asks; the owning gate stays bit-identical | occupancy/geometry: computed and written down, owning gate, timed A/B once |
| 4 | Raster group and threshold from L2 bytes | bits unchanged by the contract (`gemm32.rs:52-55`): the owning gate once; one timed A/B on the A6000, since speed is the point | integer-path reorder |

The model is held to its instrument. For each sitting, a table of predicted winner and gap against the measured ones
goes in the doc. A miss names the wrong term. No gate guards speed, so no gate pins a time.

First prediction for step 3 on sm_86 `[derived]`: plain against staged at m = 2..4 lies inside ±10 %. Staged is at
least level at m = 4 (GB10 +4 to 7 %). The sign at m = 2 and 3 is unknown. Next action: a gap outside the spread at
m = 3 makes the sm_86 row m* = 3. A level result keeps the incumbent.

## 7. Against #1151's load-time race

| | Load-time race (#1151) | Derive first (this) |
|---|---|---|
| First load | times every cell. For Qwen3.8's MXFP8 sites: 3 shapes × (3 walks at m 2..4 + 2 at 5..8) = 51 cells × ~10 ms ≈ 0.5 s `[derived]`, plus a cache file | attribute reads and occupancy queries, µs; no file |
| Later loads | reads the cache | the same arithmetic |
| Across machines | two machines with the same card can freeze different picks where the gap is inside the noise (GB10 m = 3 is "level"). Bits agree; speed reports differ | same build, driver and card give the same picks; picks differ only by card facts |
| Contention | a load timed beside a display, another process or a gate on the same card freezes a wrong pick | cannot be misled by contention |
| Routing | times the probe's routing, not the serving one | prices the expectation; one histogram checks it |
| Explains a pick | no (a timing) | yes: the terms are printed |
| Test surface | cache hit, miss, stale, corrupt and concurrent; the timer; `off` = today's dispatch | a pure golden table on the Mac; picks = fixed on tuned shapes; the forced bench as instrument |
| A card nobody has measured | times it: this is the race's real advantage | exact picks still apply; residue pairs keep the incumbent until a row arrives. The loss is bounded by the band, ≤ ~10 % on those kernels |
| A new shape | times it | computes it; the residue rows are in shape-free terms (m, waves, tiles an expert) |
| Wrong when | noise or contention at the one timing, frozen in the cache | a term is missing from the model (caught by the step-3 bench), or a row is stale (caught by md5) |

Kept from #1151: one owner beside `site.rs`; only bit-equal candidates (exact tile against MMA and HC narrow against
wide stay out); picks fixed before capture; ranks of a split model pick alone (same bits); an off switch; every pick
named in the load log and the status JSON; start with a few hot sites; the kernel-level check first (#1151 §5). The
disagreement is narrow: whether the load runs the race, or reads the row that check produced.

One part of the race is worth keeping later, outside the load path: an explicit command that runs the forced bench
on a user's card and prints a table row they can contribute. Loads stay deterministic, and an unmeasured card gets a
path to a row.

## 8. Resource timeline

- **Load.** Rule: host CPU µs; card, PCIe, DRAM and NVMe idle. Race: about 0.5 s of card time for one model's first
  load `[derived]`, the host waiting on it.
- **Decode step.** A pick changes one kernel on the engine's one stream. On a whole-card placement (Qwen3.8 on GB10, a
  dense model on one card), the step is the sum of its kernels, so a kernel's saving reaches the wall. On a hybrid
  placement, it depends on the flow. Where the card's work overlaps the host tier, a card kernel inside the host's busy
  time is worth 0. Where the two run in series, it counts in full: the 09-25 V4.1 prompt batch is 140.7 = 94.0 (host)
  + 46.7 (card) ms a layer, a sum (rig-log plan.md:87-95). The picker never changes which resource is the wall, so
  it is never a flow lever.
- **No flow lever is missed.** The candidates differ in on-chip traffic, waves and launch count, not in what overlaps
  what. The one bulk lever (one launch instead of ⌈m/8⌉, §3.3) is already today's rule.

## 9. Open questions for #1151's author

1. Do your GB10 numbers for the staged walk (`ea840911c`) and the slab entry (`e78ce30ce`) exist as bench output? If
   so, they can become sm_121's first rows once step 0's forced arms re-run them.
2. Which other sites did the commonize work find picked on one card? Each one is a row in §3.4 before it is a
   candidate.
3. Is the 5090 that took #1140's PIN available for sm_120's rows?
