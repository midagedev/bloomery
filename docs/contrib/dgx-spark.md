# Porting bloomery to the NVIDIA DGX Spark

This is the map for a contributor who owns a DGX Spark (GB10: an Arm CPU and a Blackwell GPU on one unified memory) and
wants to build, run or extend bloomery on it. It says where the port stands, how to build and check a change on a Spark
today, and which work is left and whose it is. The work is cut into phases P0 to P5 (section 1). P0, P2 and P3 are
done, and P1 is not finished.

Today a Spark builds the tree from source, runs a model that fits its memory whole on the card, and runs the fixture-tier
gates (the gates that run on small generated model files, section 4) in local mode, which runs a gate recipe on your own
machine (sections 2 and 4). What a Spark's memory and card change, and what was measured on them, is in sections 3, 5,
6 and 7.

The large work left is models larger than the memory pool (P4, section 8) and several linked Sparks (P5, section 9).
Section 10 lists every open item, and section 11 holds the maintainers' decisions. To start: build and run a first
model (section 2), then run the contributor gate list in local mode (section 4), and take an item marked for a
contributor in section 10.

**How to read the numbers.** A claim about the code cites `path:line`. Lines are at main `6b630e5e` unless the text names
another commit: the pool invariant (section 7.2) and section 8 were read at `61039d6e`, section 9 at `cce176c5`, and a
line marked "at `b289130a`" was re-read for this revision. A thing that was not seen is marked `[unverified]`, and a
number computed here is marked `[derived]`. "The contributor" is the one contributor who did the Spark work so far on
two DGX Sparks; a reading of theirs is marked `[#13]` or links its comment, with its conditions. A number that a
measurement proved wrong is struck through and stays in place, with the measured value and its comment beside it
(`AGENTS.md`). The maintainers ran nothing on a Spark.

Related documents: [`local-check.md`](local-check.md) (checking a change on your own machine: local mode and timing
mode), [`new-model.md`](new-model.md) (the other contribution track), [`sparkdec.md`](../research/sparkdec.md) (the GB10
decode step, kernel by kernel) and [`tierengine.md`](../research/tierengine.md) (the unified tier, one residency across
card, RAM and NVMe). `tierengine.md` letters its design items, among them (d) the memory kind behind the arena, (h)
unified memory and (k) N nodes, and numbers its migration steps S0 to S13, among them S7, the pinned arena, and S10, one
budget solver.

## 1. Where the port stands

| Phase | Done when | State (2026-10-13) | Left, and who |
|---|---|---|---|
| **P0** compiles on aarch64 | `cargo check` of the host chain passes on aarch64 Linux, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe` on the Spark; on x86 nothing moved: `just ptx-scan` equal and `gate-qdot`, `gate-ops` green | done: [#14](https://github.com/midagedev/bloomery/pull/14), `d95c7ec9` (design R: `qdot` refuses its host-tier types off x86, answer 1) | nothing (the site map is Appendix A) |
| **P1** fixture gates pass | the host gates whose inputs are in process or published, and the three fixture families' gates, are green or each red is a named, Arm-only clause; a first bit-gate pass on cc 12.1 | not done. In: design S of `qdot`, its scalar mirrors ([#15](https://github.com/midagedev/bloomery/pull/15), `021dab61`), local mode ([#38](https://github.com/midagedev/bloomery/pull/38)), timing mode ([#39](https://github.com/midagedev/bloomery/pull/39)), names for the Arm-only reds ([#43](https://github.com/midagedev/bloomery/pull/43)), the Arm memory-order audit ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6118769122)), `gate-engram`'s eviction clause (`86022f32`, answer 18). The contributor list on `697790c4`: 29 of 93 items green ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)); on `b289130a`, 30 of 98, and the contributor judged the host-tier protocol part closed ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6135340457)) | the V4.1 DSpark draft fixture, still unpublished (only a draft reader refuses, [#159](https://github.com/midagedev/bloomery/pull/159)). `glm5next/corpus-prose.ids`, `engram/corpus-prose.ids` and `fx_ref_glm5next_vis_taps` are on the refset dataset since 2026-10-14 ([#53](https://github.com/midagedev/bloomery/issues/53#issuecomment-6168041480)). The first cc 12.1 pass is section 6 |
| **P2** a real model runs | Clef-Flash, Qwen3-30B-A3B and Qwen3.6-35B-A3B answer from `bloomery-serve` on the Spark with a clean plan and `PoolOver` holding; a placement unit gate holds Σ card + host ≤ pool | done: [#16](https://github.com/midagedev/bloomery/pull/16), `7507942c` (section 7). The Spark run's rows are the Spark's P2 tape ([#16](https://github.com/midagedev/bloomery/pull/16#issuecomment-6103154837)) | nothing |
| **P3** Spark-specific speed | the section 5 term measured; a Spark table with its own ruler; `BLOOMERY_ARCH=sm_121` rows | done: the term ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104914439)), the table ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6107847345)), the row prefetch ([#27](https://github.com/midagedev/bloomery/pull/27), `fed530d2`), the router top-k ([#35](https://github.com/midagedev/bloomery/pull/35), `81355f2f`), `sm_121` ([#42](https://github.com/midagedev/bloomery/issues/42)), ~~`crates/threads` P/E split~~ (not needed: no decode or prefill step on the default path uses the pool, [#41](https://github.com/midagedev/bloomery/issues/41)), the cc 12.1 `ptx-shapes` tables and the cc 12.x occupancy column (section 6) | nothing |
| **P4** above the pool | a model larger than the pool (V4.1, GLM-5.3-Flash `UD-Q4_K_XL`) answers on one Spark with its overflow on the NVMe tier, the card computing the tier's experts from their slots | read rates measured (section 8). S7 is in: the pinned arena behind `BLOOMERY_NVTIER_MEM`, `device_slot` and the `devmap` tests (section 8). GLM and V4.1 load at serving defaults on one Spark and answer ([#100](https://github.com/midagedev/bloomery/pull/100), section 8) | the card leg, the decode kernels that read the experts from the arena's slots (contributor, answer 19) |
| **P5** several Sparks (×2, ×4) | a model plans and runs across N linked Sparks under the section 9 criteria, N = 2 first; the placement unit gate `nodes_scale` holds the criteria at N = 1, 2, 3 and 4 | the fabric and the protocol probe ran on two Sparks ([#44](https://github.com/midagedev/bloomery/pull/44), `de57867c`; section 9.6). The placement half's node plans and `nodes_scale` run on synthetic Sparks ([#94](https://github.com/midagedev/bloomery/pull/94), [#95](https://github.com/midagedev/bloomery/pull/95); section 9.9); section 10 lists where the gate differs from 9.9's words | the placement half's engine side: a remote node's census, the card plans each node's budget comes from, and decode at n = N (contributor, answer 20; section 10); the protocol half, on the plan the maintainer agreed to in [#127](https://github.com/midagedev/bloomery/issues/127#issuecomment-6155272828) (contributor): the `Link` seam, the counter relay (section 9.5, item 2) and the RDMA link on one machine (items 1, 3, 4 and 7) are in; node 1's tier server (items 5 and 6) and the N = 2 run are left |

Files the open work touches: P1, the unpublished V4.1 DSpark draft fixture; P4, the card's gather kernels that read the arena in
`crates/gpu/src/host/nvtier.rs` (the card leg); P5, `crates/placement` (`Machine`, `nodes_scale`),
`crates/gpu/src/host/link.rs`, `link/pump.rs` and `link/rdma.rs` (the `Link` seam, the counter relay and the RDMA end)
and the tier server (9.5, items 5 and 6).

## 2. Build and first run on a Spark

**Build.** The contributor's builds run in the `nvidia/cuda:13.3.1-devel-ubuntu24.04` container (CUDA 13.3 for aarch64,
the tested toolkit, answer 7) with LLVM 21.1.8's ARM64 release `llc` named by `CUDA_OXIDE_LLC` `[#13]`. Name it: without
`CUDA_OXIDE_LLC`, cuda-oxide picks the Rust toolchain's `llc` (LLVM 23) first (section 6). The host driver needs CUDA 13.0 or later
(`crates/serve/src/preflight.rs:16`). `.cargo/config.toml` and `.cargo/cuda-oxide.toml` carry `target-cpu=native` for
`aarch64-unknown-linux-gnu` (`f0139df3`), so a Spark needs no local flag. Device crates build with `cargo oxide` only,
for example:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2 --release --bin bloomery-serve
```

The binary carries `sm_86` PTX, and the driver compiles it for cc 12.1 at the first start, about 6.5 s on the Arm cores
`[derived]` once per `CUDA_CACHE_PATH` (section 6).

**Smallest first placement: everything on the card, no host tier, no NVMe tier.** Two ways in, both generic already:
`--place a` is `Pick::Rank(0)`, "the largest visible card" (`crates/placement/src/placement/devices.rs:236-243`), and an
unknown card takes the census path with its own total and free bytes (`devices.rs:98-123`); the Qwen3 and Qwen3.6 seat
opens "CUDA device 0" (`crates/gpu/src/lib.rs:2124`; `docs/BUILD.md:100` at `b289130a`). `--place gate` plans the 3090's bytes on the
smallest visible card with the room for them (`devices.rs:241`), which on a Spark is the GB10 with a 3090's share. The plan
reads an integrated device's room from `MemAvailable` (section 7.2), so it no longer needs a `drop_caches` before the
load, as the first run did (section 3).

Models by size, in the order to try them (`docs/models.md:12,31-34`, `docs/BUILD.md` table): Clef-Flash 4.8–9.7 GB,
Qwen3-30B-A3B `Q4_K_M` 18.6 GB, Qwen3.6-35B-A3B `Q4_K_M` 21.2 GB; then Qwen3.8 `UD-Q4_K_XL` 111.3 GB and GLM-5.3-Flash
`UD-Q2_K_XL` 101.3 GiB (the size chosen "to fit a DGX Spark", https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan-ledger.md#L359) if the first `MemAvailable` reading
allows. A 64 GB Spark (section 3; announced, not yet shipping) holds only the first three.

## 3. The machine

Every row was read from NVIDIA's own pages on 2026-10-10, by `curl` of the page and a text search for the quoted string.
A fact that no NVIDIA page stated carries no number.

| Fact | Exact text | Source |
|---|---|---|
| Superchip | "Powered by the NVIDIA GB10 Grace Blackwell Superchip" | <https://www.nvidia.com/en-us/products/workstations/dgx-spark/> |
| CUDA compute capability | table cells "12.1" then "NVIDIA GB10 (DGX Spark)" (so `sm_121`) | <https://developer.nvidia.com/cuda-gpus> |
| CPU | "20-core Arm processor (10 Cortex-X925 + 10 Cortex-A725)" | <https://docs.nvidia.com/dgx/dgx-spark/hardware.html> |
| CPU architecture | "DGX Spark is based on ARMv9.2-A architecture." | <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html> |
| CPU layout | "The cores in DGX Spark are organized as two clusters with up to 5 P-cores and 5 E-cores each." Cache lines are 64 bytes | same page |
| Compiler target | "Since LLVM version 21 and gcc version 15, it is also possible to specifically target DGX Spark with -mcpu=gb10"; "-march=native" selects it when compiling on the Spark | same page |
| NEON / SVE2 / dot-product / i8mm | not stated by NVIDIA on any page read. The SIMD page (`.../porting/simd.html`) speaks of SVE and SVE2 in general only | ~~`[unverified]`; first action on a Spark: `grep -m1 Features /proc/cpuinfo`~~ read on two DGX Sparks: the ISA row of the readings below (`asimd`, `asimddp`, `i8mm`, `sve`, `sve2` among them) `[#13]` |
| Memory | "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth" | hardware.html |
| Memory variants | "64 GB LPDDR5X* or 128 GB LPDDR5x, coherent unified system memory"; the 64 GB model is listed as "Coming Soon … exclusively through participating OEM partners" | product page |
| NVMe | "1 TB or 4 TB NVMe M.2 with self-encryption" | hardware.html |
| GPU | "NVIDIA Blackwell Architecture with 5th Generation Tensor Cores, 4th Generation RT Cores"; "CUDA Cores : 6,144"; "Copy Engines : 2" | hardware.html |
| SM count | 48 `[derived: 6,144 / 128, assuming 128 FP32 cores an SM]` | — |
| Power | "GB10 SOC Thermal Design Power (TDP) is 140W" | hardware.html |
| OS / toolkit / driver | "base operating system derived from Ubuntu 24.04"; "CUDA Toolkit CUDA 13.0"; "Launch Driver: R580.GA UDA driver" | <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/software-requirements.html> |
| Unified memory | "DGX Spark uses a unified memory architecture. On CUDA contexts the system memory returned by the pinned device memory allocators (e.g. cudaMalloc) cannot be coherently accessed by the CPU complex nor by I/O peripherals" (the page's subject is GPUDirect RDMA: "we suggest to allocate the communication buffers with the cudaHostAlloc API") | <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html> |
| Free-memory reading | "the cudaMemGetInfo API does not account for memory that could potentially be reclaimed. As a result, the memory size reported by cudaMemGetInfo may be smaller than the actual allocatable memory" | optimization.html |

How the facts meet our pins: the release needs a driver with CUDA 13.0 (`crates/serve/src/preflight.rs:16`,
`CUDA_MIN = 13000`), which is the R580 that DGX OS launched with. Our tested toolkit is 13.3 (`docs/BUILD.md:11-23`, "Other
CUDA versions … are not tested"), newer than the 13.0 DGX OS ships; whether a 13.3 aarch64 toolkit and an aarch64 LLVM 21
`llc` are available: they are, as the contributor builds in the `nvidia/cuda:13.3.1-devel-ubuntu24.04` container with
LLVM 21.1.8's ARM64 release tarball `[#13]`.

**Readings on two DGX Sparks `[#13]`** (2026-10-10; the same on both unless two values are given):

| Item | Reading |
|---|---|
| OS, kernel, driver | DGX OS 7.6.0 (Ubuntu 24.04.5), `6.17.0-1032-nvidia`, driver 595.84 (`cuDriverGetVersion` 13020), host toolkit 13.0.88 |
| GPU | cc 12.1; 48 SMs (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`); integrated 1; pageable memory access 1, through the host page tables 1; L2 24 MiB |
| Memory | `cuDeviceTotalMem` 130,661,761,024 B and 130,596,122,624 B; `MemTotal` 127,599,376 kB and 127,535,276 kB; idle `MemAvailable` 116.6–116.9 GiB; idle `cuMemGetInfo` free 115.4 and 114.8 GiB, 117.3 GiB just after `drop_caches`; 16 GiB swap |
| `cuMemGetInfo` against `MemAvailable` | a probe allocated 1 GiB chunks with 51.1 GB in page cache and stopped at its own limit, 12 GiB short of `MemAvailable`: `cuMemGetInfo` read 4.7 GB free at 68.7 GB allocated, while allocations went on to 111.7 GB as the kernel gave page cache back; `MemAvailable` at the start less what was allocated came within 0.5 GB of its end reading. `MemAvailable` is the pool's room |
| ISA | `asimd`, `asimddp`, `i8mm`, `bf16`, `sve`, `sve2`, `svei8mm`, `svebf16`, `atomics` |
| Base page size | 4096; THP `madvise`; `Hugepagesize` 2048 kB |
| Cores | cpu0-4, cpu10-14 Cortex-A725 (max 2.808 GHz, 512 KiB L2); cpu5-9, cpu15-19 Cortex-X925 (max 3.9 GHz, 2 MiB L2); no SMT |
| L3 | `cache/index3`: cpus 0-9 share 8 MiB, 10-19 share 16 MiB, the split of `cluster_cpus_list`; the pool's `detect_topology` forms two groups of ten |
| cpufreq | `scaling_cur_freq` present; driver `cppc_cpufreq` |
| `nvidia-smi` | `memory.*`, `power.limit`, `clocks.mem` are `[N/A]`; `clocks.max.sm` 3003 MHz |
| Build | `cargo oxide doctor` passes; the flat `target-cpu=znver3` made rustc crash (SIGSEGV in LLVM) on some crates; `f0139df3` (S2) gives aarch64 `target-cpu=native` |
| NVMe drive | one Spark: a 4 TB PCIe Gen5 x4 drive ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921)); another, read on 2026-10-12: a 1 TB PCIe Gen4 x4 DRAM-less drive ([#71](https://github.com/midagedev/bloomery/issues/71#issuecomment-6126758679)). Units ship different drives; their read rates are in section 8 |
| First run | Qwen3.6-35B-A3B `Q4_K_M` loads whole on the card and answers, after `drop_caches`: before it the plan read 3,762,335,744 B free, put 10,102 experts on the host tier, and the host union refused every `Q4_K` stack by name (section 7's invariant in a P0 run); the sm_86 PTX JIT-compiles for cc 12.1 on driver 595.84 |

## 4. Gates and timing on your own Spark

Gates and timed runs on your own card go through `tools/box.sh`'s local mode and its single-card timing mode.
[`local-check.md`](local-check.md) section 3 (local mode) and section 5 (timing) say how to set them up and what they
refuse. This section has only what is different on a Spark. The P1 run used one command per item, from a clean tree of
the commit:

```sh
BLOOMERY_BOX_LOCAL=1 BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_BOUND=1800' just <recipe>
```

**What differs on a Spark.**

- **Fixture paths.** `BLOOMERY_DATA` and the fixture root move by environment (`tools/ref/ref-paths.sh:67,88`), but a
  reference set's identity is the first shard's full path (`AGENTS.md`; a set's `# model` line,
  `crates/refset/src/fixture.rs:19-20`). `/models/fixtures` stays the one legal root, so on a Spark it is a symlink
  (answer 2).
- **Card names.** A Spark's card is named neither 3090 nor A6000, and local mode takes the one local card without a name
  match. `--place gate` plans the 3090's bytes on the smallest visible card with the room for them, the GB10 here, and local mode refuses the
  two-card recipes (rc 64) ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)). Which
  fixture gates open on a card with no box name is in `local-check.md` section 3. `nvidia-smi` prints `[N/A]` for
  GB10's memory fields `[#13]`; the census reads the driver API (`gpu/src/lib.rs:1789`).
- **`gate-ptx-spill`** runs on GB10 at `sm_86` and `sm_121`, against the cc 12.1 tables (section 6).

**Fixtures.** A fixture is "a model file much smaller than the real one that keeps every
per-layer shape" (`crates/model/src/fixture/mod.rs:1-4`) and "its bytes are a function of the seed and the source header
alone" (`:29-32`), pinned by per-type digests (`fixture/digest.rs:1-10`). `fixture generate <real first shard> <out dir>`
writes it (`crates/model/src/bin/fixture.rs:4-5`). Three families have fixtures (`crates/refset/src/fixture.rs:33-37`:
`qwen4exp`, `deepseek41`, `glm5next`); every other family is `self` in the table (`ref-paths.sh:75-79`) and its
gates are deferred by name as real-file. So "P1 fixture gates" means those three families plus the host gates whose
inputs are in process or published, not every `gate-*` recipe. A text classification of the `justfile` (not the batch
tool's): of 136 classified recipes, 44 are host-only and 60 take a card in both tiers, and 32 are real-tier only
(`real-only.sh` or an inline exit 66); several host gates (`gate-qdot`) read harness dumps under `$BLOOMERY_DATA`.

A contributor cannot make the **reference sets**: they are `ik` node dumps of each fixture (`dump-ref-fixture`,
`justfile:609`) from the C++ harness the maintainers build (`just build-ref`). They are public at
<https://huggingface.co/datasets/midagedev/bloomery-refsets> (17 sets, 6.48 GB, with each base model's licence). The
fixture files themselves need no download, because their bytes are reproducible: `fixture fetch` makes them from each
source model's header; their sizes on disk are in [`local-check.md`](local-check.md) section 4. `fixture generate` does not read the source's tensor data,
but it opens each source file through the loader's bounds check, which refuses a tensor that ends past the end of the
file (`crates/gguf/src/lib.rs:957` at `e659f4fe`). On qwen38
([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6117747643)), the real shards and the same shards
cut at the data offset and grown back to their full length as holes gave the same bytes, and a cut shard left short was
refused (`data at 18720+0 size 675430400 exceeds the file`); no length in between was tried. So a header grown to the
file's length as a sparse file is enough, which is what `fixture fetch` writes, at the size the repo lists for each
file (`crates/model/src/bin/fixture.rs:24-27` at `e659f4fe`). The shortest length the check passes is the end of the
last tensor, which the header's tensor offsets and sizes give `[derived]`. On aarch64 the fill matches the x86_64
pins: the digest test passes with all 12 pinned cases, and `fixture fetch glm5next` run twice wrote identical bytes,
shard 1 with the sha256 pinned as `FIXTURE_SHA256`. The generator copies the source's `tokenizer.*` and `general.*` keys
(`fixture/mod.rs:12-14`), so a fixture carries the source model's tokenizer; answer 3 says which families are published.

**What the fixture tier exercises on a Spark.** The fixture header "names the card budget that forces the plan to offload"
(`ref-paths.sh:33-35`), so the fixture gates run the **host tier**: `qdot`, the CPU attention, the card/host protocol. A
Spark contributor therefore needs design S for `qdot` at P1 (scalar mirrors, correct bits, slow), not just design R.

**Timing.** Only `tools/ref/depth-qwen3moe.sh` opts in to the timing mode today (`TIMING_LOCAL_RUNNER=1`,
`tools/ref/timing-card.sh:34`); another runner refuses `BLOOMERY_BOX_CARD=local`. On GB10 the witness reads
`power.limit` as `[N/A]` and `scaling_cur_freq` from `cppc_cpufreq` `[#13]`. The Spark's own ruler comes from an A/A of
one binary against itself, six rounds a sitting: decode 0.9987 ± 0.0065 at depth 6 and 1.0026 ± 0.0063 at depth 512,
prefill 0.9971 ± 0.0024 at P = 512 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6107847345)).
A reference engine's arm in the same lease is a build the contributor makes for aarch64; no recipe makes it.

**Evidence.** A Spark number carries CONTRIBUTING.md's evidence list and `tok/s @ n=N, depth D, card` (`AGENTS.md`). It
decides a pull request on the Spark, and it can be published in the Spark's own table with the machine named; a table
never mixes cards (`AGENTS.md`, "Public numbers name their machine"). A direct run of `cnode_probe` is enough when it
names the SM clock and the commit (answer 17).

## 5. Decode on one Spark: the cost model against the measurements

The question for the port is what a Spark is for. The decomposition below was written from repo numbers and the facts of
section 3 before any Spark run; the measured values now sit beside its predictions. [`sparkdec.md`](../research/sparkdec.md)
breaks the measured step down kernel by kernel.

**Anchor [measured, not re-measured since 0.2.3].** Qwen3.6-35B-A3B `Q4_K_M`, whole model on one RTX 3090 at its 250 W
cap: 193.9 tok/s, one request's rate (`README.md:111` and footnote 1), which is 5.16 ms a token `[derived]`.

**Terms of one decode token, whole model on the card.** Both machines read the same weight bytes a token: 1.8 GB, 3 B
active parameters (the "A3B") × 4.85 bits (21.2 GB file, `docs/models.md:33`, over 35 B parameters) `[derived]`. The
byte time is bytes / bandwidth. Each machine has its own table.

| Term, 3090 box | Value |
|---|---|
| Byte time at the memory | 2.6 ms at ~700 GB/s (`AGENTS.md`, "a grid near ~700 GB/s") |
| Rest of the step (launch chain, attention, norms) | 5.16 − 2.6 = 2.6 ms `[derived]` |
| H2D / PCIe | 0 |
| NVMe | 0 after load |

| Term, Spark | Value |
|---|---|
| Byte time at the memory | 6.7 ms at the 273 GB/s peak; ~~8.3 ms at 80 % of it `[assumed 80 %]`~~ 9.5 ms at the measured 208.8 GB/s on `b289130a`, 76 % of 273, for the gemv kernels' 1,978 MB a step (8.6 ms for this row's 1.8 GB) `[derived]` ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6136140998)). At `04d0c0ae` the same bytes took 11.1 ms at 177.5 GB/s `[derived]`; under `nsys` that gemv time splits into 8.40 ms at the plain read rate and 2.75 ms past it, and the other kernels take 1.12 ms more ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104914439)) |
| Rest of the step (launch chain, attention, norms) | ~~2.6 ms if it is latency, 4.4 ms if it scales with SMs (82 / 48 = 1.7; the 3090's 82 SMs are `[unverified]` here)~~ measured: the gaps between kernels take 0.076 ms a step ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104377342)), and the 2.75 ms past the plain read rate sits inside the gemv kernels, inside the 11.1 ms of `04d0c0ae` above ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104914439), [maintainers](https://github.com/midagedev/bloomery/issues/17#issuecomment-6105032860)) |
| H2D / PCIe | no PCIe leg for weights: the whole-card seat has no host tier |
| NVMe | 0 after load; the load reads the file once |

**Prediction [derived], from the struck terms above:** a step of 8.3 ms (bytes hide the rest) to 12.7 ms (bytes plus an SM-scaled rest), so 79–120 tok/s,
point 92 tok/s (bytes plus latency, 10.9 ms). That is 0.4–0.6 of the 3090's rate, and the byte term is 2.6–3.2 times
longer than on the box. The `machineaxes` round reached 2.47× for the A6000 (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md#L200).
Measured: 84.44 tok/s @ n=96, depth 6, GB10, on main `6854c2e9` with the
tools of #38 and #39, SM clock 2,411 to 2,515 MHz ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6107847345)), inside the band. That build is older than #27's row
prefetch, which made the step 1.6 to 1.7 ms shorter ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6105836896)).
On `b289130a`, which has #27: 99.46 tok/s @ n=96, depth 6, GB10 (the mean of six rounds) on another DGX Spark, SM clock
2,418 to 2,541 MHz, stock clocks, A/A 1.0014 ± 0.0046 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)), also inside the band.

**Resource timeline per decode token (whole on card, Spark).** Host CPU: a graph launch and the sampler, off the critical
path (no host tier), cost `[unmeasured]`. Card SMs: ~~the 2.6–4.4 ms rest, overlapped with the byte stream~~ the rest sits inside the gemv kernels (the table above). Card DRAM and
host DRAM are **one resource**, the 273 GB/s pool: the ~~6.7–8.3 ms~~ 9.5 ms byte term at the measured rate of `b289130a` (11.1 ms at `04d0c0ae`) `[derived]`. PCIe: 0. NVMe: 0. The wall is that one
resource's time (or at most bytes plus the latency part), not the sum of five. Consequences:

- Host/card asynchrony (overlapping CPU expert work under card work) is worth 0 here: there is no host expert work to hide
  and the CPU's DRAM traffic competes with the card's.
- Once the byte term is at the pool's sustained rate, a kernel change that removes instructions predicts 0
  (`AGENTS.md`, "instruction count on a kernel near ~700 GB/s: model says 0"). The levers left are fewer bytes (quant, KV
  size), launch count when the latency part is the larger one, and SM occupancy only if the rest is compute.
- A Spark's value is its 128 GB, not its bandwidth: V4.1 `Q3_K_M` (347.3 GB) and GLM-5.3-Flash `UD-Q4_K_XL` (199.7 GB)
  (`docs/BUILD.md`, model table) do not fit the pool and would need the NVMe tier; Qwen3.8 `UD-Q4_K_XL` (111.3 GB) fits
  with 13.9 GB of the idle `MemAvailable` (116.6 GiB = 125.2 GB, section 3) left for KV and runtime `[derived from #13]`.

**The one term a run must settle, with its expected value:** the sustained read rate of our decode gemv kernels on the
LPDDR5x with the CPU idle. Expected 200–235 GB/s (73–86 % of 273). Measured on `b289130a`, which has #27's row prefetch: 208.8 GB/s, the
byte-weighted rate of the gemv kernels in the decode flow (median of three runs, 208.3 to 211.4), on one DGX Spark at
stock clocks with nothing locked and a median SM clock of 2,418 MHz over busy samples ([#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6136140998)), inside the expected band. At `04d0c0ae`, before #27, the same method
read 177.5 GB/s (median of three runs, 175.6 to 179.2, the SM at 2,418 to 2,437 MHz, [#17](https://github.com/midagedev/bloomery/issues/17#issuecomment-6104914439)), and that
build read 177.8 again in the `b289130a` session. A plain read kernel measured 233 GB/s on `cuMemAlloc` memory (0.85 of 273)
and 235 GB/s on write-combined `cuMemHostAlloc` pages ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921));
~~our gemv kernels' own rate is still unread~~. Decision rule: at or
above 200 the decode flow is byte-bound and P3 kernel work predicts 0 beyond byte-saving levers; below 160, look at page
size, TLB reach and clocks (`nvidia-smi` clock-event reasons) before any kernel work. The 208.8 GB/s of `b289130a` is above
the rule's 200 line, so the first branch holds. The 177.5 GB/s of `04d0c0ae` fell in the 160 to 200 band the rule
leaves open, and [`sparkdec.md`](../research/sparkdec.md) reads that band from those runs. Everything else in this file
is settled by reading.

## 6. The card

The card runs the tree's `sm_86` PTX through the driver's JIT. `sm_121` PTX was checked on GB10 and gives the same SASS for
213 of 215 functions, so it is a question of pins, not of speed.

- **Targets today.** Recipes pass `--arch {{arch}}`, the justfile's variable (139 lines): `BLOOMERY_ARCH`, default `sm_86`,
  so `BLOOMERY_ARCH=sm_121 just <recipe>` passes `--arch sm_121`; ~~whether the pinned toolchain accepts it is `[unverified]`~~
  it does (the `sm_121` bullet below). `tools/gate.sh`, `tools/lds-scan.sh` and
  `tools/sass-scan.sh` read it through `tools/arch.sh`; `tools/ptx-scan.sh` takes `sm_86`, `sm_120` and `sm_121` (the
  bullet on pins below) and refuses any other arch. The kernels are SIMT code with no generation-specific
  instruction (the three `ptx_asm!` sites are `cvta`, `ld.global`, `max.f32`: `crates/gpu/src/linear/delta.rs:121-157`,
  `flash_gqa_prefill.rs:289`; `fault.rs:714` not read).
- **Documented state.** `docs/cuda-oxide.md:5`: "Everything here runs on Ampere (sm_86) only." The toolchain ledger
  (`docs/upstream/nvlabs-ledger.md`) has no `sm_121` or Blackwell row; its only Spark-related entry is row 32 (the flat
  `extra-rustflags`). The GB10 runs below found no defect of the pinned cuda-oxide; the one gate they saw fail was decided by the
  `llc` or libNVVM that made the code (the K-quant bullet).
- **P0–P2 need no new target.** The binary carries `.target sm_86` PTX and the driver JIT-compiles it at load; the release
  states it (`tools/release/README.release.md:15-16`: "the archive carries sm_86 PTX, which the driver compiles for the
  card on first start"), and the loader says it for Blackwell: "A payload built for a standard pre-Blackwell target, such as
  `sm_86`, may instead be converted to PTX and JIT-compiled by the driver on Blackwell"
  (`cuda-host/src/embedded.rs:58-62` at fork rev `4da6c13`, `Cargo.toml:30`), and the first run on a Spark did it
  (section 3 `[#13]`). The first JIT on the Arm cores, measured on one Spark (driver 595.84,
  [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6117747643)): a first start of `bloomery-serve`
  with Qwen3.6-35B-A3B `Q4_K_M` and an empty `CUDA_CACHE_PATH` took 10.5 to 11.8 s to `listening`, and a later start
  with the cache in place 4.1 s. The start without the cache minus the start with it, 6.4 to 7.7 s (median 6.5 s), is
  an estimate of the JIT cost `[derived]`. The first pair's difference is 1.2 to 1.3 s larger than the other two; the
  extra is not user CPU time (that start's system time was 2.1 to 2.2 s higher), and its cause was not measured. For
  comparison, in the same conditions `oxart_jit` on the binary's whole `.oxart` section (311 entries) took 8.7 to 8.8 s
  with an empty cache, counting the file read, the context creation and the attribute reads, and 0.4 s with the cache
  in place; the difference, 8.3 to 8.4 s, is the JIT of every entry by the same subtraction `[derived]`. The first
  start is inside the 10 to 25 s measured on the box (`README.release.md`).
- **`sm_121` for P3.** The fork's target table has `capability: 121` (`cuda-target-spec/src/lib.rs:382,417,447`) and its
  selection tests name `sm_121a`/`sm_121f` (`cuda-oxide-codegen/src/target/tests/selection.rs:162-179`). ~~Whether LLVM 21's
  NVPTX backend accepts `sm_121` is `[unverified]`~~ LLVM 21 emits `sm_121` PTX and the driver loads it; on GB10 the
  driver JIT gives the same SASS from `sm_86` and `sm_121` PTX for 213 of the 215 functions of `generate_qwen3moe`, the
  generated ids are the same, and a timed sitting found no decode difference outside its margin ([#42](https://github.com/midagedev/bloomery/issues/42)); `cargo oxide doctor` passes on aarch64 `[#13]`. The fork's `tcgen05` path
  is datacenter-Blackwell only (`rustc-codegen-cuda/README.md:107-108`: "need `sm_100a`, which a consumer `sm_120` lacks"),
  and we use no such instruction.
- **Pins that follow the compute capability.** `gate-ptx-spill` reads one table per PTX arch and card capability,
  because `spill` follows the arch ptxas assembles for and `jit_local` the capability the driver JITs for.
  `tools/ref/ptx-shapes.tsv` is `sm_86` on cc 8.6 (the RTX 3090). In `tools/ref/ptx-shapes/`, `sm_86-cc12.1.tsv` and
  `sm_121-cc12.1.tsv` were pinned on GB10 (driver 595.84). The recipe names the base table and that directory, and
  `tools/ptx-spill-check.sh` reads each binary against the one its scan's banner names (`arch=`, `jit-cc=`). On cc 12.1 the JIT keeps more local memory than on
  cc 8.6 for 23 rows, 22 of which have none on cc 8.6 (14 rows of `generate_ds41`, 9 of `gate_e2e`; the largest is
  `q4k_gemv_tiles`, 184 bytes a thread). The two arches give these rows the same `jit_local`, and at `sm_121` ptxas also spills on them; each nonzero
  row carries a PIN line. A kernel added on a cc 8.6 card gets its
  cc 12.1 rows from a scan on a Spark; until then `gate-ptx-spill` on a Spark names it unpinned, and
  `tools/check-ptx-names.py` reads the cc 12.1 tables for stale rows only. `tools/ptx-scan.sh`'s blk/SM column takes
  cc 12.x's limits for `sm_120` and `sm_121` (24 resident blocks an SM, where 8.6 has 16); on GB10 it equals the
  driver's `cuOccupancyMaxActiveBlocksPerMultiprocessor` for all 551 entries of the three binaries at `sm_121`.
- **Bit gates.** The memo concludes the bit gates have no generation dependence by structure (they compare against oracle
  files) and names two residual risks: `ptxas` contracting `.rn`-less mul-then-add differently, and the `ex2.approx`-class
  approximate instructions in about 30 places (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md#L198, https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/machine-axes.md#L10-L16). ~~The proof is
  one bit-gate pass on the Spark (`gate-gpu-e2e`'s count pin and the card family gates).~~ `gate-gpu-e2e` needs V2-Lite's
  ik greedy and argmax sets, which are not public, so the contributor list still defers it as real-file. The first pass
  on cc 12.1 is the fixture-tier family e2e gates ([#53](https://github.com/midagedev/bloomery/issues/53#issuecomment-6158672937),
  answer 2): `gate-gpu-glm5next-e2e` now that `glm5next/corpus-prose.ids` is published, `gate-gpu-qwen4exp-e2e` after
  Julian's tier-check fix ([#148](https://github.com/midagedev/bloomery/pull/148)), and `gate-gpu-ds41-step` after the
  `ref-paths.sh` change ([#159](https://github.com/midagedev/bloomery/pull/159)). Against the `fx_` sets they hold bands,
  an argmax inside a tie band, and exact routing ids. `gate-gpu-ds41-step` checks engram row ids, and routed expert ids
  except near ties. `gate-gpu-qwen4exp-e2e` checks the ik-routed walk's expert ids, and zero refused flips. Their
  bit-for-bit clauses compare our own runs (eager against replay, a rerun, and layer by layer). They do not show that
  cc 12.1 gives the same bits as cc 8.6. ~~The repo has only ever run cc 8.6~~ The box has only ever run cc 8.6
  (`tools/recipes.py:2498` at `3d750af3`: "both cards here are sm_86"). A green fixture pass shows no gross error.
  The logits-row diff in [#53](https://github.com/midagedev/bloomery/issues/53#issuecomment-6159465373) against a 3090 run of the same commit
  and fixture closes that bits question. `gate-gpu-e2e` on cc 12.1 can come later.
- **Device pick.** The real-tier gate plan opens the card by the names in `CARDS` and refuses any other card by name
  (`crates/gpu-gates/src/bin/shared/gate_card.rs:8-9, 38-47`: "none of this workstation's cards"). The fixture tier plans on
  `--place a`, "the largest visible card as the device reports itself" (`gate_card.rs:4-6`) and is not affected. The real
  tier is the maintainers' tier regardless (section 4).
- **K-quant bits and the `llc`.** `gate-gpu-kquant` failed when the PTX came from the Rust toolchain's `llc` (LLVM 23),
  which cuda-oxide picks ahead of LLVM 21, and passed under LLVM 21.1.8
  ([#54](https://github.com/midagedev/bloomery/issues/54)). [#63](https://github.com/midagedev/bloomery/pull/63)
  (`0d3c28ab`) spells the one-column walk's rounding, so neither `llc` nor libNVVM decides its bits. Which `llc` a build
  names is still #54's open item.
- **MXFP4 on Blackwell.** The block-scaled FP4 MMA `mma_m16n8k32_mxf8f6f4_f32_e2m1_e2m1` gives GGUF MXFP4's values bit
  for bit on GB10 at `sm_121a`, `sm_121f` and `sm_120f`, with the raw GGUF bytes and each e2m1 code in bits 5:2 of its
  byte. `sm_120a` PTX does not load on GB10 (`a PTX JIT compilation failed`; `ptxas` says a `sm_120a` program "cannot be
  compiled to future architecture"), so a Spark build names `sm_121a`, or `sm_120f` or `sm_121f` for a family build
  ([#19](https://github.com/midagedev/bloomery/issues/19#issuecomment-6104507343)). FP4 prefill stays parked
  ([maintainers](https://github.com/midagedev/bloomery/issues/19#issuecomment-6105033234)).

## 7. One pool

### 7.1 The double count

Placement was built for a discrete card, a PCIe host tier and an NVMe tier (`Machine { cards, tiers, host }` at
`6b630e5e`). On a Spark the card and the host share one memory, so the plan must count them against one pool.
[#16](https://github.com/midagedev/bloomery/pull/16) (`7507942c`) does that with `Machine::unified` (7.2), and for
several Sparks `host` and `unified` move into `nodes[0]` of a per-node `Machine` (section 9.2).

**The double count, concretely.** Three readers see one pool:
`cuDeviceTotalMem` (`crates/gpu/src/lib.rs:1789`), `cuMemGetInfo` (`:1995`, `:2302`: the card's `free_bytes`, which caps the
plan's card budget, `devices.rs:98-123`) and `/proc/meminfo` `MemAvailable` (`workstation.rs:667`, `host_room` `:858`: the
host room). A plan that checks card bytes and host bytes each against its own figure can admit up to twice the pool on a
Spark. NVIDIA adds that `cudaMemGetInfo` under-reads, because reclaimable page cache and swap are not counted (section 3), so
the card's figure is also too small right after the model file was read. The invariant needed is one: Σ card + host ≤ pool,
with the pool read as `MemAvailable` (plus what the load will drop), not `cuMemGetInfo` (item M4 of the `machineaxes` memo: extend `Machine` and
`workstation.rs`, no parallel type).

**Which terms vanish and which stay:**

| Term | On the Spark |
|---|---|
| H2D promotion over PCIe (`cuMemcpyHtoDAsync`, `crates/gpu/src/host/xstream.rs:1223,1261` at `cce176c5`; the staging rings `:283,760,1614`) | vanishes as a PCIe cost; a copy from pageable memory into a `cudaMalloc` buffer is still a DRAM-to-DRAM copy at half the pool's rate (reads and writes both cross it: 21.2 GB in ~0.16 s `[derived]`), negligible beside the file read |
| Router D2H handoff and the host service protocol (`host/mod.rs:1-60`) | unused when no layer has host experts; works on `cuMemHostAlloc` pages (`host/link.rs:293`, `xstream.rs:267`), the allocator NVIDIA recommends; correctness on Spark `[unverified]` |
| CPU expert compute | stays possible but pointless: 20 Arm cores against 48 SMs on the same bytes. It is the path the fixture tier exercises (section 4) |
| The pool's bandwidth | stays, as the single term (section 5) |
| Page cache of the model file plus the device copy | stays and doubles the footprint until dropped; `nvtier.rs:12-30` already drops pages after use (`MADV_DONTNEED`, the `release_union` path) for the NVMe tier |
| NVMe → DRAM stream for models above the pool | stays; ~~its rate on a Spark is `[unverified]`~~ its rate depends on the drive (section 8). Today the NVMe tier lends its RAM slots to the host leg (`crates/gpu/src/host/nvtier.rs:1-9`), so the CPU computes those experts; on one pool the card can read the same slots without a copy, so the card computes them (P4, section 8) |
| KV cache bytes | stay, and they come out of the same pool |

### 7.2 The pool invariant (#16)

[#16](https://github.com/midagedev/bloomery/pull/16), landed as `7507942c`, takes M4's shape cut to what P2 reads
(answer 11):

- `census()` reads `CU_DEVICE_ATTRIBUTE_INTEGRATED` into `DeviceInfo::integrated`. On an integrated device the card's
  free bytes are the host's room (`MemAvailable` and the cgroup levels), not `cuMemGetInfo`. Nothing matches on a card
  name.
- `Machine` gains `unified: Option<UnifiedPool>`. On such a machine the host is the pool itself, with no OS reserve,
  because `MemAvailable` already leaves out what the OS and other processes hold.
- `Plan::violations` gains a named `PoolOver`: the cards' totals with their margins plus the host's whole need (the NVMe
  split's arena and window included) must fit the pool. `HostNeed::cards` gives the cards their share wherever the host's
  need is read.
- On a card with its own memory nothing moves: `integrated` is false, `unified` is `None`, `HostNeed::cards` is 0.
- A plan above the pool is refused by name (`PoolOver`). A model above the pool keeps its overflow on the NVMe tier
  (section 8).

A model that fits the pool keeps its card budget. On one pool an expert costs the same bytes on either side, so a split
does not change the sum, and moving experts to the host only moves them to the slower side (the table in 7.1). The
default placement on a Spark stays "everything on the card" for such a model. A model above the pool gives up card
bytes to the NVMe tier's floor (section 8).

## 8. Models larger than the pool (P4)

A model larger than one Spark's pool keeps its overflow on the NVMe tier, and on one pool the card computes those
experts from the tier's slots in place (answer 12). The read rates that choice rests on are measured (8.1). With S7 the
arena on a Spark is write-combined pinned pages by default (`BLOOMERY_NVTIER_MEM` unset or `auto`; `anon` keeps the
anonymous mapping), and the card leg is the contributor's next piece (answer 19). P4 is done when GLM-5.3-Flash
`UD-Q4_K_XL` or V4.1 `Q3_K_M` answers on one Spark with the card computing from the slots.

**The load at serving defaults ([#100](https://github.com/midagedev/bloomery/pull/100)).** Both models load at serving defaults on one Spark and answer. The NVMe
tier's floor is the least host room a plan with that tier needs, and on one pool the cards' share of the pool is one of
its terms. A card filled up to the pool's reading left the room under that floor, so both plans were refused.
`placement::plan_in_pool` now plans such a model again under a lower card budget until the floor holds. A card budget
the user gives (`BLOOMERY_CARD_BUDGET`) is planned as given, and a card with memory of its own plans as before.

**No arena at serving defaults.** Those default plans give the tier no RAM arena (`arena=0 arena_slots=0` in the `plan`
line). Their room is above half of the host leg's routed-expert bytes, and there the split dial (`nvme_arena_of`) pages
the overflow through the file mapping instead. So the default load reads no slot, and S7's pinned arena and the card
leg are not on its path. How a plan on one Spark gets an arena is in section 10.

**Today, on a machine with PCIe.** The NVMe tier's RAM arena is one anonymous mapping carved into slots, filled by
`O_DIRECT` reads and "lent to the host leg through the model side's `TierSlots` seam" (`crates/gpu/src/host/nvtier.rs:1-9`):
the CPU computes those experts with `qdot`. On a PCIe machine that is the right flow, because the CPU reads host DRAM in
place and a copy to the card would cross the slower bus.

**On one pool.** The card can address pageable memory, an anonymous mapping included, through the host's page tables
(section 3: `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` 1, "coherently accessing pageable memory without
`cudaHostRegister`", and `..._USES_HOST_PAGE_TABLES` 1 `[#13]`; the attributes' text is the CUDA driver API reference's).
So the card can compute the experts in the arena's slots where they lie, with no copy. The attributes promise no rate
and no residency: a slot page the kernel swapped out (16 GiB swap, section 3), compacted or migrated costs the card a
translation fault that the kernel serves. The arena must hold its pages (`mlock` or an equal guard), and the rule below
measures the rate. The CPU computes nothing on the decode path, which matches section 5's timeline: 20 Arm cores beside
48 SMs on the same 273 GB/s add only contention.

**The owner.** The unified tier ([`tierengine.md`](../research/tierengine.md)): one residency across the card, RAM
and NVMe. On an integrated device its card and RAM levels are one pool, so the residency has two levels, the pool and
the NVMe drive. Its slot seam hands a filled slot's address to the card leg instead of the host leg. Its design for an
integrated device is `tierengine.md` (h). S7 there (the pinned arena, `device_slot` and the `devmap` tests, run on a
GB10 and an RTX 5090) is in, and the card leg that reads from the slots is the contributor's (answer 19).

### 8.1 What was measured

**The read rate, against its decision rule.** One card gemv kernel reading the same expert bytes from (i)
`cuMemAlloc` memory, (ii) `cuMemHostAlloc` memory (the reading asked for on #13) and (iii) an anonymous mapping as the
arena allocates it, with the CPU idle. The suspect term is translation reach: the arena's pages are 4 KiB today, since
THP is in `madvise` mode `[#13]` and `AnonMap` is memmap2's `map_anon` with no `MADV_HUGEPAGE`
(`crates/engram/src/direct.rs:145-150`). Expected: (iii) under (i) ~~by the 4 KiB translations~~, (ii) near (i)
~~`[predicted, no Spark number exists]`~~. Measured with a kernel that does a decode step's gate_up reads without the
arithmetic, repeat reads, the median of 4 runs, outside the lease runner (exploratory readings,
[#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921)): (i) 233.3, (ii) 223.6 (199.4 on the first read after a fill) and with
`WRITECOMBINED` 235.1, (iii) 157.4 GB/s; (iii) with 2 MiB pages reads 159.2, so translation reach is not the cause. The rule follows the copy's cost: a copy into card memory moves 2 × the slot's
bytes through the same pool once per fill (the table in 7.1), and the slower read is paid on every use, so with h
uses per fill the card reads in place while 1/r(iii) − 1/r(i) < 2 / (h · r_pool) `[derived]` (h = 1: down to a third
of (i); h = 10: at 0.83 of (i) or above).

With these rates (iii) is outside the band for every h above 3, and `MADV_HUGEPAGE` does not bring it inside. (ii) with
`WRITECOMBINED` reads at (i)'s rate on first and repeat reads, so it is inside the band for every h, and `O_DIRECT`
reads land correctly in those pages at the drive's rate (11.20 GB/s against 11.16 into an anonymous mapping, the same
comment). So on an integrated device the arena moves to `cuMemHostAlloc` with `WRITECOMBINED`, and the card reads its
slots in place ([maintainers](https://github.com/midagedev/bloomery/issues/13#issuecomment-6104493159)). The rule's other branches (huge pages, a copy on the copy engines) are not taken.

**A slot hit.** A decode kernel gathers random slot rows, not whole buffers. On one Spark, today's `_sel`-shaped load
form reads random slot rows of a WC arena at 167.8 GB/s (6 interleaved rounds, slots no launch had read since the
fill), and whole-line loads on 128-B aligned parts at 241.4 GB/s (the prefetch loop off, the arena touched once after
its fill, 3 rounds on another Spark) ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288), row 1; loads only, no decode). One `cuMemHostAlloc` with `WRITECOMBINED` of 64 GiB took
5.288 s, linear in size (row 2). How these readings price a hit and the arena's allocation is `tierengine.md` (h) and
its S7 row, as the maintainers answered ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134411281)).

**The drive.** ~~The NVMe read rate on a Spark is `[unverified]`.~~ The NVMe read rate on a Spark depends on the drive: a 4 TB PCIe
Gen5 drive reads 11.16 GB/s with 1 MiB `O_DIRECT` preads at queue depth 16 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921)), and a 1 TB PCIe Gen4
DRAM-less drive about 4.0 GB/s with about 1 MB expert parts at random offsets at queue depth 8 ([#71](https://github.com/midagedev/bloomery/issues/71#issuecomment-6126758679)). Under one load, expert parts read by `probe_nvread` at queue
depth 16, the two drives read 8.4 to 8.9 and 5.8 GB/s ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288), row 3). Our box's drive reads 6.77 GB/s at queue depth 16
(`docs/cards/qdsweep.card`); that number belongs to that drive and does not carry over. Above the pool the decode is
bound by the tier's misses, whose flow model is the unified tier's. Reads of expert parts at queue depth 3, the unit a
fill uses, ran at 4.6 to 5.5 GB/s on these two drives ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288), row 3; `tierengine.md` (j) takes
that curve).

## 9. Several Sparks (P5)

P5 runs one model across N linked Sparks, N = 2 first, and no code may assume N = 2 (answer 13). The design came from
a read-only round (2026-10-11). Since then the fabric and the protocol probe ran on two Sparks with one direct cable
(9.1, 9.6). The figures in 9.7 and 9.8 are still `[derived]` from section 5's cost model, the repo's byte counts and the
assumptions stated once in 9.7. No engine code has run on two Sparks.

The work splits in two halves, as proposed in [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134316241) and accepted in answer 20. The
placement half needs no Arm protocol: `host` and `unified` into `nodes[0]`, the per-node shape of 9.2 and the unit gate
`nodes_scale` (9.9). It is the contributor's and does not wait for S10. The protocol half is the counters that cross
the link, the `Link` seam, the transport of 9.5 and the N = 2 run. It starts when the contributor judges P1's protocol
part closed, which they did on `b289130a`, starting with the `Link` seam ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6135340457)). The
contributor asked on that comment to carry the crossing counters, which are the module owner's (the maintainers'),
and the maintainer agreed in [#127](https://github.com/midagedev/bloomery/issues/127#issuecomment-6155272828) to the counter relay, which carries them on the link and
leaves the cards' operations as they are (9.5, item 2).

Plans are judged by eleven criteria (9.8). Only criterion 1, every node's need within its own pool, refuses a plan.
Criteria 2 to 7 are objectives that the planner meets or names why it cannot, criterion 8 is gate clause 1, and criteria 9
to 11 are held by runs and review. The derived tables of 9.7 and 9.8 use inputs that measurements have since replaced,
and `nodes_scale`'s derivation script recomputes them (section 10).

### 9.1 Link facts

Read by `curl` on 2026-10-11, with a text search for each quoted string. The two rows marked "user" are forum posts by
users, not NVIDIA.

| Fact | Exact text | Source |
|---|---|---|
| Ports and rate | "Each DGX Spark has two QSFP ports (sometimes called “ConnectX-7 ports”) on the back of the device. Each port provides up to 200 Gigabits per second (Gb/s), but the incoming speed is also determined by the cable that you use." | <https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html> |
| Ports, hardware page | "2x QSFP Network connectors (ConnectX-7)" | <https://docs.nvidia.com/dgx/dgx-spark/hardware.html> |
| Ethernet only | "The QSFP ports support Ethernet configuration only." (so RoCE, not InfiniBand: `[derived]` reading) | spark-clustering.html |
| NIC to SoC | "The NIC connects independently to the two external QSFP ports, and it connects to the SoC through two independent PCIe Gen 5 x4 links." | spark-clustering.html |
| ×2 direct, ×3 direct, ×4 with a switch | "It supports up to three DGX Spark systems connected directly through cables, and up to four systems when using a switch." | spark-clustering.html |
| ×2: one cable | "One QSFP cable for direct 200GbE connection between two devices"; "Full bandwidth can be achieved with just one QSFP cable." | <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/connect-two-sparks/README.md> |
| ×3: a ring, both ports | "1. Node1 (Port0) to Node2 (Port1) 2. Node2 (Port0) to Node3 (Port1) 3. Node3 (Port0) to Node1 (Port1)"; "Each CX7 port provides full 200GbE bandwidth." | <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/connect-three-sparks/README.md> |
| ×4: a switch, one port a node | "Configure four DGX Spark systems for high-speed inter-node communication using 200Gbps QSFP connections through a QSFP switch."; "QSFP switch with at least 4 QSFP56-DD ports (at least 200Gbps each)"; "One cable per spark" | <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/multi-sparks-through-switch/README.md> |
| No GPUDirect RDMA on GB10 | "the system memory returned by the pinned device memory allocators (e.g. cudaMalloc) cannot be coherently accessed by the CPU complex nor by I/O peripherals like PCI Express devices (e.g. the Mellanox NIC). Hence the GPUDirect RDMA technology is not supported, and the mechanisms for direct I/O based on that technology, for example nvidia-peermem (for DOCA-Host), dma-buf or GDRCopy, do not work." | <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html> |
| What to use instead | "for Linux RDMA applications based on the ib verbs library, we suggest to allocate the communication buffers with the cudaHostAlloc API and to register them with the ib_reg_mr function." | cuda.html |
| Rate NVIDIA measured, one rail each | `ib_write_bw` at 65536 bytes: "92.57" and "97.28" Gb/s on the two RoCE devices of one cable, "Total throughput = 92.57 + 97.28 = 189.85 Gbps". The run printed "Number of qps : 1", "TX depth : 128", "Mtu : 1024[B]" and "PCIe relax order: ON": a pipelined stream with relaxed ordering on, not one message in flight | <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md> |
| One rail, about 100 Gb/s (user) | "TCP tests or IB tests on a single sub-interface will max out at ~100 GBps, but NCCL is able to use both and achieve full 200G." The unit is Gb/s: the same sentence says "full 200G", and NVIDIA's rows above read 92.57 and 97.28 Gb/s a rail | <https://forums.developer.nvidia.com/t/369673.json>, a user reply |
| One rail alone (user) | "Rail 1 alone: 109.28 Gbit/s"; "Both rails, Gigabyte to ASUS: 98.02 + 98.02 = 196.04 Gbit/s" (two GB10 boxes of other makers, one cable) | <https://forums.developer.nvidia.com/t/376298.json>, a user report |
| Latency | the guide names the `ib_write_lat` and `ib_read_lat` commands and prints no value. No source gives a latency for a direct link | `[unverified]` |
| Write order without relaxed ordering | "relaxed ordering has the following impact: RDMA write-after-write message order is no longer guaranteed"; "a completion still ensures that all data is visible, including data from prior transfers." | <https://raw.githubusercontent.com/linux-rdma/rdma-core/master/libibverbs/man/ibv_reg_mr.3> |
| Arm ordering | "some reorderings cannot happen on x86_64 but can happen on ARM"; "Multithreaded applications however must explicitly make use of memory barriers to ensure correct operation." | <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/memorder.html> |
| What NVIDIA runs on linked Sparks: tensor parallel | two Sparks, vLLM: "Start the vLLM inference server with tensor parallelism across both nodes." with `--tensor-parallel-size 2`; TRT-LLM: "Qwen3-235B-A22B (two Sparks only)" with `--tp_size 2`; four: "If you are running with four or more sparks, you can comfortably run this model with tensor parallelism." | <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/vllm/README.md>, <https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/trt-llm/README.md> |

Derived from the rows above:
- **A node's NIC cap is fixed by its SoC links, not by its port count `[derived]`.** Both ports reach the SoC over the
  same two Gen 5 x4 links (2 × 15.75 GB/s raw a direction `[derived: 32 GT/s a lane, 128b/130b]`). One cable measured
  189.85 Gb/s = 23.7 GB/s a direction, and one RoCE device with one QP saw 92.57 Gb/s = 11.6 GB/s. A ring of three (both
  ports) and a switch do not raise the cap. Every node, the stage node included, has about 23.7 GB/s a direction
  whatever N is.
- One decode message (at most 131 KB, 9.7) rides one rail. A prompt batch can stripe over both rails.
- A GPU cannot post to the NIC: without GPUDirect, the paths where a kernel rings the doorbell (IBGDA, NVSHMEM) rest on
  the peer mapping that cuda.html says does not work (that no such path works on GB10 is `[unverified]`). A host thread
  posts every send.

Measured on two DGX Sparks with one direct cable, one RoCE device a side (RoCE v2, MTU 4096, perftest 6.20; [#44](https://github.com/midagedev/bloomery/pull/44),
[#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288) row 5 and [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)). These are a contributor's readings, kept apart from NVIDIA's rows above.
The `-t 1` and `-t 128` rows and the second `ib_write_lat` row come from one session after the EC, SoC and VBIOS update
of 2026-10-13, with 100,000 iterations a run (#44's bandwidth runs used 500,000). Their direction means of three runs
are `[derived]`. In the perftest rows, strict ordering means the tool printed `PCIe relax order: OFF`: `ib_write_bw`
takes `--disable_pcie_relaxed` for it, and `ib_write_lat` has no option and prints `OFF`.

| Reading | Value |
|---|---|
| `ib_write_bw`, 64 KB, one rail, both directions, relaxed ordering on (perftest's default) | 109.15 to 109.30 Gb/s |
| `ib_write_bw`, 3 MiB, one rail | 109.04 Gb/s = 13.63 GB/s |
| `ib_write_bw`, one write in flight (`-t 1`), strict ordering, both directions | 24.7 and 25.2 Gb/s at 16 KB, 39.7 and 40.1 at 32 KB, 56.6 and 57.6 at 64 KB (direction means of three runs `[derived]`): 5.20 to 5.30, 6.54 to 6.61 and 9.10 to 9.27 µs a write `[derived]` ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)) |
| `ib_write_bw`, 128 writes in flight (`-t 128`), strict ordering, 16 to 64 KB | 108.71 to 109.26 Gb/s (direction means of three runs `[derived]`) ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)) |
| `ib_write_lat`, one way (`t_typical`), strict ordering, before the 2026-10-13 update ([#44](https://github.com/midagedev/bloomery/pull/44)) | 4.80 to 4.85 µs at 16 KB, 8.98 to 9.12 µs at 64 KB (the range of two runs a direction; no 32 KB run) |
| `ib_write_lat`, one way (`t_typical`), strict ordering, after the update | A to B: 5.00 µs at 16 KB, 7.03 at 32 KB, 9.43 at 64 KB. B to A: 5.03, 6.86 and 9.56 µs (direction means of three runs `[derived]`, [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)) |
| `all_gather_perf` busbw, 8 GiB | 13.78 to 13.82 GB/s out of place, 13.91 to 13.92 GB/s in place |
| `link_probe` round trip λ_probe (16 KB out, 64 KB back, strict ordering) | p50 29.76 to 30.27 µs, 0 stale lines; the engine's λ is 37 to 60 µs `[derived]` in #44 |

### 9.2 The shape

A node is one unified machine (section 7). A remote GB10 does what a tier card does today: it computes routed experts
in place from a handoff and returns rows. `Device::Card(i)` over `all_cards()` already names such a card, so the second
node is a node, not a new device kind. It cannot be a tier with a link cost on today's `Machine`: #16's `PoolOver` sums
every card of `all_cards()` into one pool, so a remote tier would be charged to node 0's pool, and node 1's host-side
rows and arena to node 0's host.

The owner shape as designed is below. Part of it is in the tree. `Machine::nodes` holds each node's `host` and
`unified`, this process's node first ([#94](https://github.com/midagedev/bloomery/pull/94)). `Fabric`, and `LinkCost` for
the `Link` below, are in `crates/placement/src/placement/nodes.rs` ([#95](https://github.com/midagedev/bloomery/pull/95)),
on `Nodes` beside the machine and not on `Machine`. `LinkCost` carries λ only, and the read rate, the NIC and the ruler
sit on `Nodes` for every node alike. `Card` has no `node` field and `Node` no `nic` yet:

```rust
pub struct Card { /* … */ pub node: usize }       // an index into Machine::nodes; 0 is this process's node
pub struct Machine { pub cards: Vec<Card>, pub tiers: Vec<Card>,
                     pub nodes: Vec<Node>,        // nodes[0]: #16's `host` and `unified` move here
                     pub fabric: Fabric }
pub struct Node { pub host: Host, pub unified: Option<UnifiedPool>,
                  pub nic: u64 }                  // bytes/s a direction, the SoC side (9.1: about 23.7 GB/s)
pub enum Fabric { Alone,                          // N = 1: every machine today
                  Pair { link: Link },            // N = 2, one cable
                  Ring { link: Link },            // N = 3, both ports
                  Switch { link: Link, hop_ns: u64 } } // N = 4: NVIDIA's ×4 is switched
pub struct Link { pub rail: u64 /* bytes/s, one message */, pub round_trip_ns: u64 }
```

- **`host` and `unified` live in `nodes[0]`** ([#94](https://github.com/midagedev/bloomery/pull/94)), read through
  `Machine::host` and `Machine::unified`. With one node in the list the plan is exactly section 7's. Today's card plan
  refuses a machine of more than one node by name (`PlacementError::Nodes`).
- `CardOver` stays per card. The node plans hold `PoolOver` once per node: each node's budget is the card plan of that
  node alone, and its routed ids are the node plan's split (9.9). `HostNeed` gains no field and is evaluated per node.
- **Node 1's figures come from a remote census.** The census reads only this process's CUDA devices. Node 1's `CardSpec`
  and its `MemAvailable` reading must reach node 0 over the control channel before `plan_tiers` runs.
- **One rule term, common to every family.** A tier takes "each layer's next ids after the stage's prefix"
  (`crates/placement/src/placement/workstation.rs:352-362`). Over a link the rule spreads the hot ids to reach the read
  share s that balances `Sh + s·R = λ·BW + (1 − s)·R` per layer (terms in 9.7). A, Sh and R come from the plan's own
  tensor bytes and roles, so no model supplies them. The node plans in the tree split by id count, as if every id were
  used alike; the router's counts that spread the hot ids reach the split when the engine side calls it. On the box's
  PCIe the same formula holds with the lane's rate, so it is one owner for both machines, not a Spark branch.
- **With the unified tier.** The second node is a compute site with its own residency, not one more tier in the read
  path: reading a remote V4.1 expert costs ~~16.8 MB / 11.6 GB/s = 1.4 ms, against 16.8 MB / 219 GB/s = 77 µs~~ 16.8 MB / 13.63 GB/s = 1.2 ms, against
  16.8 MB / 208.8 GB/s (section 5) = 80 µs to compute it in place `[derived]`. The tier's ledger gains a node column, (node, id) to state. Node 1 runs its own books over
  its own pool, arena and file copy. On a unified node a card slot and a host segment collapse (section 8), so a node's
  states are pool, arena and file. One new move is pool of node 0 to pool of node 1: a link copy of an expert (~~0.7–1.4
  ms at 11.6 GB/s a rail~~ 0.6 to 1.2 ms at the measured 13.63 GB/s a rail, 9.1 `[derived]`) on a background lane, at a step boundary, rare. The router's counts go to node 1 once a window for its arena
  eviction and drive the cross-node balance. The unified tier's per-node byte ledger, plan-time rule and prefetch rounds
  are the ones this touches. None needs to change now.

### 9.3 Three ways to split a model

L is the layer count, `t_hop` one message's one-way latency on the link. Per decode token a message carries a hidden
vector or a few rows of them, tens of KB, so the link's bytes are small and its latency is the term `[derived]`. A
prompt is different: its bytes over the link grow with P, which is why today's prompt path moves the experts to the
activations (`crates/gpu/src/host/xstream.rs:5-19`) and no expert ever crosses the link in (a).

| Split | Attention and KV | Per decode token | Per prompt batch of P = 4096 tokens `[derived]` | What scales with N |
|---|---|---|---|---|
| (a) remote expert tiers: plan (b′) (`crates/placement/src/placement/workstation.rs:352-362`): one stage node runs every layer and the head, each other node holds and computes a share of the experts, the link in place of PCIe | on the stage node | about 2·L messages to each remote node, all through the stage node's NIC (today's tier protocol per tier layer: the handoff image out and the rows back, with a go and a done word, `crates/gpu/src/host/tier.rs:6-11`) | per layer the image out (P × n_embd values: q8_1 for V4.1, f32 for GLM and Qwen3.8) and the rows back (one f32 row of n_embd per remote slot, about k/2 a token); out + back a batch: V4.1 11.0 GB (back 10.1), GLM 14.1 GB (back 11.3), Qwen3.8 12.1 GB (back 10.1) | capacity, and the expert bytes each node reads; the stage node's NIC carries every message, so it can become the max resource as N grows |
| (b) pipeline by layers | each node holds its layers' KV | N − 1 messages | P × 16 × n_embd B (the 4 hc streams in f32) at each of the N − 1 node boundaries: V4.1 336 MB, GLM 268 MB, Qwen3.8 168 MB | capacity only: one request still walks the nodes in turn; several requests fill the pipeline |
| (c) expert-parallel: every node computes its share of each layer's experts in parallel | every node runs attention on the same hidden state, so the KV cache is held N times; without that, (c) is (a) | about 2·L messages per node pair, spread over the NICs | per layer every node's partial sums, P × n_embd × 4 B, to each other node | capacity and the per-node byte term, if the expert bytes are balanced |

Decode on one Spark is bound by the pool's bytes (section 5). (a) and (c) split those bytes over the nodes and add the
latency term; (b) adds latency and splits no bytes for one request. (d), tensor parallel, is what NVIDIA documents: 9.4.

### 9.4 Recommendation, and tensor parallel

The rates below are 9.7's `[derived]` bands at their first inputs (BW 219 GB/s, the box's NVMe, the assumed ρ), not
recomputed with the measured values.

1. **(a), the star, for one request: plan (b′) with the second Spark as the expert tier and the link in place of PCIe.**
   - It keeps the bits. The join already reads one f32 row per routed slot from the tier
     (`crates/gpu/src/host/tier.rs:20-24`), so moving those bytes over RDMA moves no float.
   - Every family already has a tier path: `TierAct::Q8` for V4.1 (`crates/gpu-deepseek41/src/body.rs:2613`),
     `TierAct::F32` for GLM (`crates/gpu-glm5next/src/tier.rs:461`) and Qwen3.8
     (`crates/gpu/src/arch/qwen3moe/tier38.rs:563`). It needs no new device kind.
   - **The gain comes from fit, not from a second memory bus.** Per decode layer (a) halves only the FFN bytes;
     attention stays serial on node 0. GLM (fits two Sparks, not one): 17.5–24.9 tok/s against 3.9–8.3 on one,
     NVMe-bound. V4.1 (fits neither): the NVMe share of routed reads falls from about 25–40 % to 2–7 % and the rate from
     3.5–5.6 to 15.6–29.1 tok/s, on assumed hit rates. Qwen3.8 (fits one): 1.05–1.10× at one request, and the real gain
     is room, node 0 dropping from about 114 to about 75 GB, which frees context.
2. **(b), pipeline by layers, as a second mode on the same node shape**, for several requests in flight: with two
   requests it ties or edges (a), and it is the better prompt split (1.78 × S₁ at N = 2). It fits the shape as two stage
   cards with layer ranges on two nodes (`cards` is already in stage order, and plan (b) staged V4.1 on both PCIe
   cards).
3. **(c) is left out.** Two stage cards that each run all layers for different requests and share experts are a replica
   kind, which the shape does not have.

**Tensor parallel, TP = 2, is what NVIDIA documents** (9.1: vLLM `--tensor-parallel-size 2`, TRT-LLM `--tp_size 2`).
`[derived]` `t = (A + Sh + R)/(2·BW) + L·m·λ` with m sync points a layer. At m = 2 on two nodes it reads GLM 25–36 tok/s
and Qwen3.8 49–71 tok/s at one request, which beats (a) where A dominates. It shards every family's attention and dense
kernels (an MLA latent cache replicated), changes the float sum order (a row-parallel all-reduce; column-only splits
avoid that but take 2–4 gathers a layer), and needs an NCCL-class all-reduce every layer. It is the ceiling to remember,
not the first step: a per-family rewrite against "Common code first". Whether to take it instead is the maintainers'
decision.

### 9.5 Transport

**Raw verbs (rdma-core) under our own word protocol, with one host pump thread per node.** NCCL is the fabric check
(`all_gather_perf` per NVIDIA's NCCL playbook), and the fallback only if the probe of 9.6 shows our pump path slower
than NCCL's point to point. Without GPUDirect NCCL stages through host bounce buffers with its proxy thread `[derived
from the cuda.html line; NCCL internals unverified]`, and a send or receive kernel per message would occupy SMs. The
handoff image already lives in `cuMemHostAlloc` pages (`crates/gpu/src/host/link.rs:293`), which is what 9.1's "what to
use instead" asks for, so the NIC reads the same LPDDR with no copy. The protocol stays `host/mod.rs`'s alone
(`crates/gpu/src/host/mod.rs:56-58`), behind a `Link` seam (`crates/gpu/src/host/link.rs`). It has three impls: the
shared page (PCIe and one Spark, identity), the loopback (two pages on one machine, a host pump thread between them,
picked by `BLOOMERY_TIER_LINK=loopback`) and the RDMA link (`BLOOMERY_TIER_LINK=rdma`). The RDMA link is the
loopback's two pages, each held by its own end with its own pump thread and queue pair, the two queue pairs connected
on the machine's first RoCE v2 port (`crates/gpu/src/host/link/rdma.rs`). An end writes only its own page and reaches
the other through RDMA writes, so a node holds one end of a link between two nodes. ~~A raw-verbs build adds
`libibverbs`, whose licence under `just deny` is `[unverified]`.~~ The verbs crate (`crates/verbs`) loads `libibverbs`
at run time with `dlopen`, so no build links it (9.6, item 5).

What `crates/gpu/src/host/mod.rs` and `tier.rs` need to cross a node:
1. **Two pages, one a node.** Every word keeps one writer node. A message is one chain of RDMA writes on one queue
   pair: the bytes the counter publishes, then for a done the tier's progress, then the generation into an inbox word
   on the reader's page. The inbox and the staged values a post sends sit in a tail past the tier page's own bytes, so
   the shared page keeps its layout.
2. **The counters cross as generations; the cards keep their operations.** A row's go counter is added to by the stage
   and taken back by the tier, and its done counter the other way: two writers, at different times. A mirrored absolute
   write would erase the remote take back. A captured wait also keeps its value fixed on every replay
   (`crates/gpu/src/graph.rs`, `HostFlags`), so a wait for a target that rises each replay cannot be captured. So the
   counter is split across the two pages. On the writer's page only the writing card adds one to it, and that value is
   a generation. The writer's end posts each new generation to the reader's inbox, after the bytes it publishes. The
   reader's end adds one to the counter on the reader's page for each new generation in its inbox, and the reading
   card waits for at least one and takes it back, as on the shared page. Both ends hold the step rule: a step of 0
   carries nothing, a step of 1 carries one, and any other step publishes nothing and stops that end by name, since a
   row has at most one go or done in flight. The host's progress word on the stage's page counts the dones the
   writer's end posted and goes after their bytes, so it never runs ahead of them. A release stops all of the tier's
   counters on both ends before it raises any reader's counter, since a released card runs on and steps the counters it
   writes. Each counter's reset takes its generations as carried and resumes it alone, each end resetting only its own
   side of it, and a link stopped by name fails the reset, so the host tier stays poisoned
   (`crates/gpu/src/host/link/pump.rs`). On the shared page nothing changes: the same words, values, card operations
   and graphs.
3. **A pump thread a node.** The card cannot post to the NIC. Each end's pump load-acquires the generation (Arm needs
   an explicit acquire: the memorder row) and checks its step before any byte of the message moves. The card's
   system-scope barrier before go (`crates/gpu/src/host/mod.rs:20-24`) orders the image before the word.
4. **Data before flag across the NIC.** The pages are registered without `IBV_ACCESS_RELAXED_ORDERING`, so
   write-after-write order holds (the `ibv_reg_mr` row), with the generation last on the same QP. Only the last write of
   a message is signaled; a completion with an error stops the end by its status name, which the lost tier's detail
   carries. Write-with-immediate, where the remote pump writes the flag after the completion, was the other way: one
   more hop, receives to keep posted, and a second ordering rule. NVIDIA's perftest ran with relaxed ordering on, so its
   bandwidth is not protocol-mode bandwidth.
5. **The remote is another process.** Node 1 runs a tier server: it loads its experts from its own copy of the file,
   captures its graphs from the go order it is sent at load, and launches a replay on a "replay r of chain c" message,
   which the first image can carry (today the stage process launches the tier's graph, `tier.rs:37-42`). Before the
   plan, the same channel carries node 1's census to node 0.
6. **Loss.** The go deadline already names a lost tier (`tier.rs:49-54`, `CardLost`). A down link also means the stage
   cannot write `RELEASE` to node 1's words, so node 1 needs its own deadline that drains its streams.
7. **Faults and sequence.** The tier's fault-word copy rides before its signal (`tier.rs:42-44`). A doubled image is
   refused by the relay's step rule (item 2): a repeated generation delivers nothing, and a step above one stops the end
   by name. A stale image, bytes their generation overtook, is what item 4's order prevents, and `link_probe` counts it
   on the link's own ends: every 64-byte line carries its generation, and each card counts the lines that do not.

### 9.6 First step, and what it measured

**Before two nodes run the engine.**
1. Single-node P1 is the precondition: ~~the host-tier protocol's correctness on Arm is still open (section 6)~~ the
   host tier's Arm memory order is audited ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6118769122)), but the contributor gate list is not green
   (section 1). Two nodes add a protocol layer on top of it.
2. #16 landed with its doc line (`7507942c`). Moving `host` and `unified` into `nodes[0]` (a move-class change, before
   any remote node), the shape of 9.2 and `nodes_scale` are the placement half (answer 20).
3. The crossing counters (9.5, item 2), the counter relay in `host/link.rs` and `link/pump.rs`: a protocol change in
   the protocol half, on the plan the maintainer agreed to in [#127](https://github.com/midagedev/bloomery/issues/127#issuecomment-6155272828). The maintainer takes it on
   the structural proof of #96 (`ptx-scan` identical, the graph node count, eager = replay, the e2e set identical) in
   place of a same-lease A/B on the box (same comment). `weekly-gpu-ds41-lost` runs the owning gate's clauses over the
   loopback (`BLOOMERY_TIER_LINK=loopback`) on one card.
4. A `Link` seam where the tier page is written and read: a move-class change, `ptx-scan` identical.
5. ~~Decide the `libibverbs` dependency under `just deny`.~~ Decided in #44's review: the verbs code loads the system's
   `libibverbs.so.1` with `dlopen`, so no crate links it ([#44](https://github.com/midagedev/bloomery/pull/44)). The
   tier link's RDMA end is its second user, so it is its own crate, `crates/verbs`.

**The contributor's first step needs no engine code.** It is done: `link_probe`
([#44](https://github.com/midagedev/bloomery/pull/44), `de57867c`) and the fabric runs, with the results in 9.1. On two Sparks, one cable, one rail, under a prediction card
(`tools/ref/card.py` format; their own table, the machine named). The probe takes N from the command line from the
start, so nothing in it assumes a pair.
1. **Fabric.** `ib_write_bw -s 65536` and `ib_write_lat -s 16384` and `-s 65536` on one RoCE device, both directions.
   Then `all_gather_perf` per NVIDIA's NCCL playbook.
   - Expected bandwidth 92–109 Gb/s a rail (11.6–13.7 GB/s): NVIDIA's 92.57 and 97.28 and the user's 109.28 (9.1). Below
     90 Gb/s a rail `[a decision line, not a source: the 90 Gb/s line of the N = 4 switch check below, applied to the
     direct link]` at perftest's default of 128 writes in flight, the cable, MTU or ordering setup is wrong: fix it
     before the probe (one write in flight reads far less, 9.1). Measured: at the band's top (9.1).
   - Expected one-way latency as perftest prints it: 4–10 µs at 16 KB and 8–14 µs at 64 KB `[derived from an assumed 3–8
     µs base]`. Above 10 µs at 16 KB, λ's band of 25–70 µs is low by twice the excess `[derived]`: redo 9.7 before
     trusting criterion 2. Measured: inside both bands (9.1).
2. **The protocol probe** (a small bin, contributor-sized).
   - Node 0's kernel writes a 16 KB image and a go generation into a `cudaHostAlloc` page registered with `ibv_reg_mr`
     (strict ordering). A pump posts one RDMA write. Node 1's stream waits on its page (greater-or-equal), runs a null
     kernel and writes 64 KB of rows and a generation. Its pump posts back, and node 0's stream wait passes.
   - 10⁵ iterations, p50 and p99. Every 64-byte line of the image carries the sequence, and the probe counts the stale
     lines seen after the flag.
   - **Expected λ_probe p50 = 18–40 µs `[derived]`.** The engine's λ is λ_probe plus 7–30 µs: the tier graph's copy-in
     and copy-out nodes, the stage's handoff, go and wait memops and the join. That gives the 25–70 µs of 9.7. Expected
     stale lines: 0. Measured: inside the band, with 0 stale lines (9.1), so (a) holds.
   - Decision rule on λ_probe, against the break-even λ* per model (9.7: Qwen3.8 122 µs, GLM 556 µs, V4.1 410–490 µs):
     - **at most 40 µs** (engine at most 70 µs): (a) as predicted; Qwen3.8 1.05–1.10× at one request; GLM and V4.1 are
       decided by fit.
     - **40–120 µs**: (a) only for the models that do not fit one Spark (GLM, V4.1). Qwen3.8 sits near its λ* of 122 µs,
       so it stays on one node.
     - **over 120 µs**: the pump or wake-up path is wrong; fix it before any engine work.
     - **stale lines over 0**: switch to write-with-immediate and measure again. The verbs crate no longer has the
       write-with-immediate and receive paths (item 4), so this needs them written again.
3. Then the single-node term section 5 names (sustained BW), which sets every band above. Measured: section 5.

**What carries from N = 2 to N = 4 unchanged:** the pump threads, the strict-order registration, the relay's step rule
and the stale-line check, the pair λ, each node's NIC cap (the same SoC links) and the per-node census over the control
channel.

**What N = 4 adds:**
- **A switch health check first.** `ib_write_bw` through the switch at 90 Gb/s a rail or more, before anything else. The
  switch's own settings can hold a link far under line rate, and no source here measures a switched Spark
  `[unverified]`.
- **The switch hop**, measured as λ_switch − λ_pair. Expected +0.5–2 µs `[assumed]`.
- **A fan-out and fan-in mode in the probe.** Node 0 posts to 3 peers and takes 3 rows messages. Expected λ_fanout =
  λ_pair + 2 µs of posts + 3 × 61 KB / 23.7 GB/s ingress (7.7 µs) ≈ λ_pair + 10 µs `[derived]`.

### 9.7 Reference: bands `[derived]`

**Assumptions, stated once.** Every derived figure below holds under the first inputs stated here; the measured values
now written beside them are not fed back into the tables.
- **BW**, the pool read of our kernels: ~~219 GB/s, the 80 % of 273 GB/s that section 2 assumes (band 200–235)~~
  measured 208.8 GB/s on `b289130a` (177.5 GB/s on `04d0c0ae`, before #27; section 5). A plain read kernel measured 233 GB/s on `cuMemAlloc`
  memory (#13); ~~our kernels' own rate, section 2's one term, is unread~~.
- **λ**, the exposed round trip a layer, link and protocol: 25–70 µs. Terms: two one-way RDMA trips of 3–8 µs `[assumed:
  no direct-link source]` (measured one way: 9.1); send serialization 0.5–1.4 µs and rows serialization 4.4–11.3 µs at 11.6 GB/s (51–131 KB);
  two pump wake-ups of 1–3 µs `[assumed]`; two stream-wait passes of 2–5 µs `[assumed]`; the tier graph's copy nodes
  (`c_node` 0.85 µs, [rig-log plan.md:71](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md#L71);
  on GB10 0.553 µs, the fastest of three direct runs of `cnode_probe --time` (0.553 to 0.557), SM clock 2,372 to
  2,424 MHz, [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6128568288) row 6, on main `23ee2100` ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984))). The one measured protocol number in the repo is the host-tier bridge floor,
  21–36 µs a layer on PCIe ([rig-log plan.md:78](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md#L78)).
  Measured on one direct cable: λ_probe and the engine's λ `[derived]` are inside the band (9.1).
- **U**, the usable pool a node for model, KV and scratch: about 115 GB `[assumed]`. Section 3 reads 116.6–116.9 GiB
  (125 GB) of idle `MemAvailable` on a Spark `[#13]` and #16 reads it at plan time, so U is a cautious planning figure.
- **NVMe**: 6.77 GB/s, our box's drive at queue depth 16 (`docs/cards/qdsweep.card`). ~~A Spark's drive rate is
  `[unverified]`.~~ A Spark's drive rate depends on the drive (section 8).
- N = 4 adds a 1.5 µs switch hop and 1 µs for a further post `[assumed]`.
- The non-byte "rest" of a step (~~launch chain and latency, 2.6–4.4 ms a token~~; measured inside the gemv kernels,
  section 5) is not in the bands. It lowers
  every absolute figure by about 5–15 % and moves the ratios by under 5 % `[derived at the GLM and Qwen3.8 midpoints]`.
- V4.1 and GLM hit rates on one node (ρ₁ 0.6–0.75 and 0.75–0.9) and on two (ρ₂ 0.93–0.98) are `[assumed]`. The V4.1 and
  GLM bands are only as good as they are. For V4.1, a chat route trace recorded on a DGX Spark gives two rows ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124845802)): a fixed set
  ranked by other requests' counts, ρ₁ 0.736 and ρ₂ 0.960, inside the bands, and a shared LRU that keeps its state from
  one request to the next, 0.943 and 0.994, above them. The replay scores the 5,793 generated-token positions of 24 held
  requests on 38 of the 40 layers, with 154 and 320 experts a layer as proxies for these rooms. Neither row charges for a
  fill or an eviction, and neither is the engine's own shape (a fixed set plus a per-layer LRU of arena slots).
- The (a) figures come from a Monte Carlo of the per-layer window: each routed slot lands on node j with the ownership
  probability the rule picks, and the window is the max over the nodes of `Sh + slots·e` (node 0) or `λ + hop + posts +
  rows ingress + slots·e` (a remote). Node 0's share is optimized: 0.55–0.72 at N = 2 for Qwen3.8 and 0.07–0.15 at N = 4
  for GLM and V4.1. The script moves into `tools/` beside the gate (9.9). The gate's cells and the (a) rows at the
  gate's inputs below come from `tools/flow/nodes_scale.py` instead, which sums over the count tuples of a token's slots
  rather than sampling them (9.9).

**Model terms** `[derived]` from each model's tensor table (file sizes: `docs/models.md`):

| Term | V4.1 `Q3_K_M` | GLM-5.3-Flash `UD-Q4_K_XL` | Qwen3.8 `UD-Q4_K_XL` |
|---|---|---|---|
| MoE layers L | 40 | 42 | 48 |
| `n_embd` | 5120 | 4096 | 2560 |
| experts / top-k k / expert ff | 384 / 6 / 2304 | 288 / 8 / 2048 | 512 / 10 / 640 |
| bytes an expert | 16.77 MB | 15.20 MB | 3.17 MB |
| R, routed bytes read a token (k × e × L) | 4.03 GB | 5.11 GB | 1.52 GB |
| Sh, shared and dense FFN a token | 0.8–1.5 GB | ~~1.4 GB~~ 1.60 GB: shared expert 1.12, dense FFN of layers 0 to 2 0.48 (a header read, [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)) | 0.14 GB |
| A, attention, norms, router and head a token | 3.2–5.8 GB | 4.7–8.2 GB; the header read gives 7.36 GB: attention 6.45 (linear-attention q, k and v 3.64, attention output 2.00, MLA 0.59, the rest 0.23), head 0.67, router 0.20, the rest 0.04 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)) | 3.4–4.4 GB |
| file; fits | 347.3 GB; neither one nor two Sparks | 199.7 GB; two, not one | 111.3 GB; one, marginal |

**The per-layer model.** A layer is `attention → router → handoff → [node 0: shared + its routed share ‖ node 1: λ + its
routed share] → join`. The next layer's attention needs the join, so node 1 idles through node 0's attention.
- (a) per token: `t = A/BW + max(Sh + s·R, λ·BW·L + (1−s)·R)/BW`. With s balanced, `t = A/BW + (Sh + R)/(2·BW) + L·λ/2`.
- **Routing imbalance.** A node owns a fixed set of experts, and a token's k slots fall on the two nodes about
  Binomial(k, ½). The join waits for the larger half: E|X − k/2| is 0.94 slots at k = 6, 1.09 at k = 8 and 1.23 at k =
  10, which is 2.87 ms (V4.1), 3.19 ms (GLM) and 0.85 ms (Qwen3.8) a token. With two requests the union averages it
  down. In a prompt it is negligible. The lever that removes it: copy the hot experts to both nodes and pick the node
  per token (Qwen3.8 has the room; the handoff kernel must write places per token).
- Gain over one node, when the model fits it: `ΔT = L·(F_l/BW − λ)/2 − L·Δ_l/BW`, with `F_l = (Sh + R)/L` and Δ_l the
  per-layer imbalance. The break-even per-layer round trip is `λ* = (F_l − 2·Δ_l)/BW`: **Qwen3.8 122 µs, GLM 556 µs,
  V4.1 410–490 µs** at the first BW, 219 GB/s, not recomputed.
- (b) with one request in flight: the wall is the sum of the nodes plus the hops. With two in flight, the max.
- (c) with one request equals (a) at a fixed s = 0.5, and both nodes carry the dense weights. With two requests,
  attention is data-parallel (one request a node) and the all-to-all runs both ways: `t = (A + Sh + R(2)/2)/BW + L·λ`,
  with R(2) the routed bytes of the two tokens' union.

**Capacity** (pools N × about 115 GB):

| N | V4.1 (needs about 268 GB resident) | GLM (about 203 GB) | Qwen3.8 (about 114 GB) |
|---|---|---|---|
| 1 | no: ~~about 154 GB of routed on NVMe~~ node 0's pool has room for 154 of the 384 routed ids at 4,096 positions and 153 at 32,768, so 230 (155.0 GB) and 231 (155.7 GB) do not fit | no: ~~about 88 GB on NVMe~~ 128 of 288 ids on NVMe, 82.4 GB | yes, marginal |
| 2 | no: about 43 GB on NVMe ((b): about 19 GB a node) | yes, ~~about 25 GB spare~~ every id held: the pools have room for 160 and 176 of the 288 ids | yes |
| 4 | yes, about 190 GB spare ((b) about 67 GB a stage; (c) adds 6 GB of dense a node) | yes ((c) adds 11.4 GB of dense a node) | yes |

The cells that replace the struck figures are `[derived]` from the files' headers (V4.1 `Q3_K_M`, GLM `UD-Q4_K_XL`)
at `cb5688c9`, with node 0's card plan of the file and a pool a node of the 115 GB that 9.7 assumes, the same at 4,096
and 32,768 positions where no position count is named. GLM's are the plans of `plan_nodes`
([#95](https://github.com/midagedev/bloomery/pull/95)). The node plans refuse the V4.1 file at every N: node 0's card
plan leaves its engram tables (84.5 GB) on the NVMe tier and reads 5,280 B of them a token, and `Terms::of` refuses
such rows by name (9.9, Terms). V4.1's cell at N = 1 comes from the room node 0's budget (`NodeBudget::of`) leaves for
routed ids: the pool the plan reads from those 115 GB, 115.28 GB, less its fixed bytes (10.86 GB at 4,096 positions,
12.14 GB at 32,768), over 673.9 MB an id.

For Qwen3.8 at N = 2 under (a), node 0 holds about 75 GB (4.5 non-routed + 28.8 PLE + about 39 routed + KV) against
about 114 GB on one node, and node 1 about 40 GB.

The tables below and the "fires" table of 9.8 are `[derived]` from the assumptions above as first written: BW 219 GB/s,
NVMe 6.77 GB/s and the assumed ρ. They are not recomputed with the measured inputs.

**Decode, tok/s, one request / N requests in flight (total):**

| Split | N | V4.1 | GLM | Qwen3.8 |
|---|---|---|---|---|
| one node | 1 | 3.5–5.6 (NVMe) | 3.9–8.3 (NVMe) | 36.1–43.3 |
| (a) star | 2 | 15.6–29.1 (with NVMe) / 21–40 | 17.5–24.9 / 28.4–37.4 | 37.8–47.8 / 65.7–80.4 |
| (a) star | 4 | 25.8–39.2 / 70.3–91.7 | 19.3–29.2 / 54.1–70.7 | 38.8–50.7 / 122.7–151.2 |
| (b) pipeline | 2 | 10.7–21.1 / 21–42 | 14.9–19.6 / 29.8–39.1 | 36–43 / 72–87 |
| (b) pipeline | 4 | 20.0–26.1 / 80–105 | 14.9–19.6 / 59.5–78.1 | 36–43 / 144–173 |
| (c) EP, data-parallel attention | 2 | about (a) / 35.1–46.1 pool only, the NVMe term on top | about (a) / 26.8–35.2 | about (a) / 62.0–78.3 |
| (c) EP | 4 | about (a) / 65.6–84.6 | about (a) / 50.7–65.6 | about (a) / 119.7–149.8 |

- One node with two requests in flight, for comparison: Qwen3.8 58.5–67.6, GLM 4.4–9.6, V4.1 3.8–6.0 tok/s in total.
- (b) with one request is the one-node pool time plus N one-way hops, whatever N is.
- For V4.1 the NVMe term dominates the one-node and N = 2 columns. The ratio is only as good as ρ₂. The unified tier's
  prefetch is the lever that hides it.
- For Qwen3.8, A is 67–73 % of the bytes, so (a) gains 1.05–1.10× at one request on two nodes.

**Prompt, P = 4096, in units of S₁** (one node's in-pool prompt rate at this P, a rate of a card and model pair;
~~**S₁ is unmeasured on a Spark**~~ a first reading on one Spark: 1,038.3 and 1,054.5 tok/s for Qwen3.8 `UD-Q4_K_XL`
prompts of 4,087 and 4,092 tokens at `v0.3.0`'s serving defaults, MTP draft on, [#46](https://github.com/midagedev/bloomery/issues/46#issuecomment-6116080838)) **and the absolute cap
that binds:**

| Split | N = 2 | N = 4 | Cap that binds |
|---|---|---|---|
| (a) star | V4.1 1.47, GLM 1.43, Qwen3.8 1.17–1.22 × S₁ | 1.92, 1.82, 1.28–1.37 × S₁ | Amdahl on node 0's attention, `1/(1−φ+φ/N)` with φ the FFN share of the active parameters (V4.1 0.64, GLM 0.60, Qwen3.8 0.29–0.36); then node 0's NIC ingress, `nic / (L·k·(N−1)/N·4·n_embd)`: V4.1 9,644, GLM 8,610 and Qwen3.8 9,644 tok/s at N = 2; 6,429, 5,740 and 6,429 at N = 4 |
| (b) pipeline | 1.78 × S₁ | 2.91 × S₁ | the bubble, `N·m/(m+N−1)` with m = 8 ubatches; the NIC never binds (at most 1.0 GB a batch) |
| (c) EP, N prompts at once | about N × S₁ less the all-to-all | same | each node's NIC: 7,527 / 4,305 / 4,822 tok/s a node at N = 2 and 5,018 / 2,870 / 3,215 at N = 4 (V4.1 / GLM / Qwen3.8) |
| one node, NVMe-bound | — | — | V4.1 at most 178 tok/s (154 GB at 6.77 GB/s = 22.7 s a batch); GLM at most 315 (13 s). At N = 2, V4.1 at most 1,280 (43 GB over two drives) |

- **The fit term is larger than any of these.** One Spark reads the routed set it lacks from NVMe on every batch,
  because P = 4096 touches nearly every expert: V4.1 about 154 GB, 23 s at 6.77 GB/s, against about 3.2 s on two Sparks
  under (a) (about 43 GB over two drives); GLM 88 GB, 13 s, against 0 on two.
- (a)'s link time a prompt token on both rails is 104 µs (V4.1, Qwen3.8) to 116 µs (GLM). It sits under a prompt's
  compute when it overlaps per ubatch: the box's fastest P = 4096 prompt, Qwen3.8 on the A6000, is 1,673 tok/s = 598 µs
  a token (`README.md`), and a Spark is not faster `[assumed]`.

**Link bytes and messages per decode token, one request:**

| Split | N = 2 | N = 4 | Who carries it |
|---|---|---|---|
| (a) star | V4.1 3.08 MB, 80 messages; GLM 3.99 MB, 84; Qwen3.8 2.45 MB, 95 | V4.1 4.98 MB, 211; GLM 6.84 MB, 237; Qwen3.8 3.98 MB, 250 | all on node 0's NIC; per layer at most 2·min(k, N−1) messages |
| (b) pipeline | one boundary (16·n_embd B: 82, 66, 41 KB) and a token id | N − 1 boundaries and a token id | spread: each NIC one in, one out |
| (c) EP | as (a) with uniform ownership | as (a) | spread: each node at most 4(N−1) a layer |

**(a) at the gate's inputs** `[derived]` by `tools/flow/nodes_scale.py --table` (9.9): BW 208.8 GB/s, the 23.7 GB/s
NIC, a 115 GB pool a node, the gate's synthetic shapes (GLM's A and Sh from the header read), λ from 25 to 70 µs and A
from low to high, one request. N is the Sparks offered; the plan may leave some of them idle (criteria 2 and 7 of 9.8),
and node 0's share is the split's, not an even one. Decode carries no NVMe term, so for a model that pages it is the
pool's time only. The prompt is the Amdahl term above with the slowest node's FFN at the split, and the NIC cap is node
0's rows ingress at the split's remote share. Messages count a handoff and its rows for each remote a token reaches.

| (a) star at the gate's inputs | N | Nodes used | Decode, one request (tok/s) | Prompt (× S₁) | Node 0's NIC ingress cap (tok/s) | Link bytes a token (MB) | Messages a token |
|---|---|---|---|---|---|---|---|
| V4.1 | 2 | 2 | 22.0 to 32.7 (pool only: pages 62 to 67 ids) | 1.28 to 1.40 | 7,846 to 9,140 | 3.40 to 3.84 | 79 to 80 |
| V4.1 | 4 | 4 | 24.5 to 38.1 | 1.49 to 1.75 | 4,895 to 6,031 | 6.00 to 7.07 | 203 to 218 |
| GLM | 2 | 2 | 17.8 to 18.1 | 1.28 to 1.29 | 7,005 to 7,216 | 3.97 to 4.07 | 84 |
| GLM | 4 | 4 | 19.9 to 20.5 | 1.48 to 1.52 | 4,551 to 4,825 | 6.85 to 7.17 | 237 to 240 |
| Qwen3.8 | 2 | 2 | 36.2 to 45.9 | 1.09 to 1.16 | 10,396 to 15,055 | 2.06 to 2.77 | 94 to 96 |
| Qwen3.8 | 4 | 2 to 3 | 36.2 to 47.6 | 1.09 to 1.22 | 7,968 to 15,269 | 2.03 to 3.93 | 94 to 187 |

The rest of the decode, prompt and link tables above is not recomputed. The one-node and NVMe rows need ρ and the
drive's rate, (b) a pipeline model, (c) the union R(2), and N requests in flight a batched step; none of them is an
input of the script (section 10).

**One GLM decode layer under (a), one request, N = 2 (midpoints), the resource timeline:**

| Resource | Busy | Note |
|---|---|---|
| Node 0 pool and SMs | 0.70 ms attention (154 MB) + 0.38 ms window (shared + s·R) | serial: the attention is on the critical path |
| Node 1 pool and SMs | 0.33 ms (its routed share) | idle through node 0's attention: 31 % busy |
| Link | 16 KB out and 64 KB back: 1.4 + 5.6 µs of wire + 2 × 3–8 µs | under node 0's window when λ < λ* |
| Host CPU | a pump on each node, about 2 µs a layer each | a spinning core, as the host tier's service is today |
| NVMe | 0 | everything in the pools |

The wall is A plus the max of the two FFN halves, not the sum of the five resources. Flow levers checked: **asynchrony**
(stagger two requests so node 1 works during node 0's attention) loses to batching them, because node 0 would read the
attention weights twice (a GLM layer for two requests: 1.70 ms staggered, 1.34 ms batched); **bulk** is already one
message each way a layer; **SIMT** over the batch is the existing columns form (`Chain::Cols`). The only lever left on
the serial A is tensor parallel (9.4).

**Which resource is the max, and where each split stops scaling:**
- **(a), one request.** The floor is `A/BW + L·(λ + e/BW)`: the attention is never split, and the routing imbalance
  keeps the window from falling as 1/N (the FFN window gains only 1.1–1.5× from N = 2 to 4). Latency: `L·λ` inside the
  FFN window, half-shadowed at N = 2; at N = 4 node 0 keeps few slots (about 0.1 for GLM and V4.1), a remote is always
  the max and the term is fully exposed: 1.2–3.1 ms.
  - Qwen3.8 stops at N = 2: N = 4 adds 2.6–6 %.
  - GLM gains 10–17 % at N = 4; its ceiling at the A/BW floor is 27–46.5 tok/s.
  - V4.1 keeps scaling to N = 4 because the NVMe term leaves; its floor is 38–68 tok/s.
  - The star does not stop on the NIC in decode (at most 131 KB a layer, 5.5 µs of serialization against layers of 0.4
    ms or more). It stops on the NIC only in prompt: (N−1)/N of the rows arrive at node 0, so its cap falls from about
    9.6k to 6.4k tok/s.
  - The max resource at every N is node 0's pool on the serial attention. In the FFN window it is a remote node's
    imbalance plus λ. In prompt it is node 0's attention compute, then its NIC ingress.
- **(b)** never scales one request, and each node adds a hop (at most 40 µs in all). It scales throughput linearly while
  the requests in flight are at least N, and the prompt as `N·m/(m+N−1)`. It stops when m < N. One request: the sum of
  all pools; N requests: the slowest stage.
- **(c)** scales throughput with N requests, but every node re-reads the dense weights, the all-to-all latency is
  exposed (`L·λ`, 1.0–3.4 ms), and each node's NIC caps its prompt (2,870 tok/s a node for GLM at N = 4).

### 9.8 Reference: scaling criteria

One numbered list. Criteria 1–8 are numeric conditions that the unit gate of 9.9 holds on synthetic machines at N = 1,
2, 3 and 4; criteria 9–11 are held by runs and review. The predictor is `NodePlan::predict() -> Prediction { step_ns,
link_ns, per_node: [NodeLoad { pool, window_ns, nic_permille, msgs_per_layer, fills_milli }] }`, on the plan that
`plan_nodes(&Nodes, &Terms, &[NodeBudget], Workload)` makes (`crates/placement/src/placement/nodes.rs`). It reads node
0's card plan's tensor bytes and roles (`Terms::of`), the nodes' read rate (BW) and NIC and the fabric's `LinkCost`, and
needs no model file and no per-model code. The workload carries the prompt's S₁ (a rate of a card and model pair, so on
the workload, not the card spec), whether every id must be held, and the NVMe fills. The step is decode at n = 1;
decode at n = N is not predicted yet (section 10). The thresholds come from a machine parameter, `ruler`: the
useful-gain line is 2 × the machine's 95 % interval. The box's default is 2 % (A6000, SD 0.6 %); a Spark's own A/A
sets its value (section 4). Our ruler is never baked in.

**Only criterion 1 refuses a plan**, as `CardOver` and `HostOver` do. Criteria 2–7 are **objectives** (6 is a structural
assertion): the planner chooses which nodes it uses, the ownership and later the prompt mode to meet them, and where it
cannot it names why (`NodeIdle { node, why }`, a prompt-mode note). An unset `--place` on four Sparks still plans. Criteria
8–11 carry over from the earlier list and are not in S1–S6 of the design round.

1. **Pool (S2-i), the one hard refusal.** Every node's planned need is at most its pool: `PoolOver` per node. Today's
   `PoolOver` charges every card to one pool.
2. **Link share (S1), objective.** The exposed link term is at most 10 % of the predicted step; drop nodes until it
   holds. N = 2 to 4 gains 3–17 % of the step under (a), and a link term above 10 % cancels it where A dominates.
3. **Fill (S2-ii), objective.** The remote nodes' fills are within 10 % of each other, and node 0 keeps its KV reserve.
   Uneven fills move heat to one remote.
4. **Balance (S3), objective.** The max over the nodes of the predicted FFN read time is at most (1 + 2.5 × ruler) times
   the rule's optimum. The window is 25–50 % of the step at N = 2, so 5 % of it is 1.25–2.5 % of the step.
5. **NIC (S4), objective.** Every node's NIC is busy at most 50 % of the decode step and of the predicted prompt batch;
   where it is not, the prompt mode moves (a later (b) mode, or a remote weighted sum where its float class is
   accepted). At ρ = 0.5 the M/D/1 mean wait is half a 64 KB message, at most 3 µs, inside λ's band. A star of remote
   tiers around one stage node loses this first, at the stage node's NIC, in prompt.
6. **Messages (S5), structural.** Per layer at the busiest node: at most 2·min(k, N−1) for (a), 2 for (b), 4(N−1) for
   (c), never O(k·N); messages × `c_post` (1 µs `[assumed]`) at most 2 % of the layer. One message per slot would be 30
   posts a layer for Qwen3.8 at N = 4. A unit assertion.
7. **Never slower (S6), objective.** Each node used gains at least 2 × ruler (2 % on the box) over the plan without it;
   otherwise `NodeIdle { node, why }`. A smaller gain cannot be confirmed by an A/B.
8. **N = 1 plans exactly as a single Spark** (gate clause 1).
9. **Prefill.** `pp tok/s @ P = 512 and 4096` at each N is measured and judged like decode, in the machine's own table;
   the prompt predictor of criterion 5 is checked against it. It is a run criterion, not a unit-gate clause.
10. **Host threads.** One pump thread a node posts every send, because no GPU path to the NIC works on GB10 (9.1, 9.5).
    Its cost, two wake-ups of 1–3 µs a layer, is inside λ, so criteria 2 and 6 hold it. No other host thread stands on
    the per-layer path. This replaces the earlier "no host thread" wording.
11. **A lost node ends the request by name**, as a lost tier card does today (`tier.rs:49-54`): the go deadline names
    it, and node 1's own deadline drains its streams (9.5, item 6). No failover.

**Where each fires** `[derived]` from 9.7's first inputs (BW 219 GB/s, NVMe 6.77 GB/s, the assumed ρ), one row per model, N in {1, 2, 3, 4}, λ in {25, 70} µs, A low and high, and the prompt
at S₁ in {1,000, 3,000} tok/s (the pins of the gate in 9.9):

| Model | Decode step (ms), N = 1 → 2 → 3 → 4, λ 25 / 70, A low | Criterion 2 fires | Criterion 7 fires | Criterion 5 fires (prompt) |
|---|---|---|---|---|
| Qwen3.8 | 23.1 → 20.9 / 21.9 → 20.1 / 21.4 → 19.7 / 21.2 | N = 4, λ 70, A low (2.34 ms exposed = 11 %) | N = 3 at λ 70 with A high; N = 4 in every cell (node 3 gains under 2 %) | N ≥ 3 at S₁ 3,000 (busy 51–64 %) |
| GLM | 51.2 → 40.3 / 41.3 → 36.3 / 37.6 → 34.3 / 35.9 | never | never | N ≥ 3 at S₁ 3,000 (77–95 %) |
| V4.1 (pool only) | 38.2 → 30.1 / 31.1 → 27.0 / 28.3 → 25.5 / 26.9 | never | never | N ≥ 3 at S₁ 3,000 (72–90 %) |

Nothing fires at S₁ = 1,000. At N = 2 nothing fires except GLM's prompt, which sits on the 50 % line (4,286 against
8,610): the gate's boundary case, at ≤ and not <. The predictor in the tree reads the measured BW and node 0's real split,
and differs from this table in criterion 5's column; 9.9 gives its cells.

**Where each fires at the gate's inputs** `[derived]` by `tools/flow/nodes_scale.py --table` (9.9): BW 208.8 GB/s, the
23.7 GB/s NIC, a 115 GB pool a node, GLM's header read, node 0's NIC at the split's own share, and the gate's `RULER`,
10 ‰ (question 1 of [#95](https://github.com/midagedev/bloomery/pull/95) asks whether that is the box's ruler). An N
alone names every λ and A at it.

| Model | Decode step (ms) at the gate's inputs, N = 1 → 2 → 3 → 4, λ 25 / 70, A low | Criterion 2 fires | Criterion 7 fires | Criterion 5 fires, S₁ 1,000 | Criterion 5 fires, S₁ 3,000 |
|---|---|---|---|---|---|
| V4.1 | 38.4 → 30.6 / 31.4 → 27.7 / 29.0 → 26.3 / 27.7 | never | never | never | N = 2, λ 25, A low; N = 3; N = 4 |
| GLM | 67.4 → 55.2 / 56.2 → 50.9 / 52.3 → 48.7 / 50.3 | never | never | never | N = 2; N = 3; N = 4 |
| Qwen3.8 | 24.2 → 21.8 / 22.8 → 21.0 / 22.3 → 21.0 / 22.8 | N = 4, λ 70, A low | N = 3, λ 70, A high; N = 4, λ 25, A low and high; N = 4, λ 70, A high | never | never |

Two cells sit on a line within the change one id moved makes. V4.1 at N = 2, λ 25, A low has its busiest NIC at
501 ± 5 ‰ at S₁ 3,000, on the 500 ‰ line, and criterion 5 fires there. Qwen3.8 at N = 3, λ 70, A low has its third
node's link share at 99,956 ± 78 ppm, on the tenth, and the node is used. The cells file names each as an edge in every
row of a ruler and S₁ it is derived at, four rows in all, and the gate pins its side (9.9).

### 9.9 Reference: the gate, `nodes_scale`

`nodes_scale` is the placement half's gate (answer 20), built with the shape of 9.2. It is a set of unit tests in
`crates/placement/src/placement/nodes_scale.rs`, a pure crate, so `just mac-test` and the box's lib gate both run it. It
runs the node plans of `crates/placement/src/placement/nodes.rs` ([#95](https://github.com/midagedev/bloomery/pull/95)).
Where the gate holds a clause differently from its words below, the clause says how, and section 10 lists it.

- **Machines.** The test helper `workstation::spark_nodes(n, layers, fabric)` builds synthetic GB10 nodes. Node 0 is
  today's machine of a GB10 that the census reads with a 115 GB pool free (`workstation::gb10`, through
  `spec_of_device`); each other node is that pool. Every node reads at BW 208.8 GB/s (section 5) and has a 23.7 GB/s
  NIC. N = 1 is `Alone`, 2 `Pair`, 3 `Ring` and 4 `Switch`. `LinkCost` (the `Link` of 9.2) is set at λ = 25 and 70 µs
  and the ruler at `workstation::RULER`, 10 ‰.
- **Models.** Three synthetic shapes carry 9.7's terms for V4.1, GLM and Qwen3.8 (L, `n_embd`, k, e, A low and high,
  Sh), each term rounded down to whole Q4_K rows; GLM's A and Sh are the header read's 7.36 and 1.60 GB. Clause 1 also
  plans the Qwen3.8 file's tensor table for its first two layers, by its program's routed rule. No model file is read.
- **Terms.** `Terms::of` reads node 0's card plan: each tensor's rows the plan reads a token, in the card's format on
  a card (a BF16 tensor counts its f32 bytes) and in the file's on the host. Routed experts count their file bytes, the
  bytes the split holds them at. A row table left on the NVMe tier is refused by name, since the predictor has no fill
  term for it. The row width is an input, the model's embedding length.
- **Expected sets.** The table of 9.8, pinned from a derivation script that moves into `tools/` beside the gate with its
  own self-test, so the pins have one derivation. ~~The script is not in the tree yet. Until it is, the gate pins the
  predictor's own cells, which match 9.8's table in criteria 2 and 7 and differ in criterion 5 (clause 5).~~ The script
  is `tools/flow/nodes_scale.py`, with no code or method shared with the planner: node 0's share in real numbers by
  9.2's rule, found by bisection, and each window an expectation over the count tuples of a token's slots, built one
  slot at a time. It reads `crates/placement/tests/nodes_scale_inputs.tsv`, one row per input with its kind (measured,
  derived or assumed) and source, and writes the 144 cells (a model, N, λ and A, at the gate's `RULER` and S₁ 1,000 and
  3,000 and at a 4 % ruler and S₁ 3,000) to `nodes_scale_cells.tsv` beside it; `--table` prints 9.8's table at the
  gate's inputs and 9.7's (a) rows, and `--check` holds the file and both doc tables to them. Each value carries the
  largest change one id moved between two used nodes makes, because the planner splits whole ids. Where that band holds
  a line (the NIC's half, the link share's tenth, twice the ruler), the cell names an edge, the gate pins its side by
  hand, and the planner's value is held to the band besides. `--self-test` runs hand-worked cases, the refusals and
  `--check`; `just check-recipes` runs it.
- **Clauses:**
  1. `n1_is_today`: every N = 1 plan equals today's plan byte for byte. The gate holds it where node 0 holds every id:
     node 0's bytes are today's pool need, or both plans refuse the model by the same bytes. Paged, it holds node 0 to
     today's terms before their NVMe split and every id its pool has room for, because the node plans do not plan one
     Spark's paged model as today's plan does (section 10). The step and node 0's window are the shape's bytes over BW.
  2. `pool_per_node`: criterion 1 refuses V4.1 whole at N = 2 and places it whole at N = 4. No other machine is refused.
     A node is used while the pools before it cannot hold every id. A node whose own budget passes its pool, or that
     the final split gives no id, is named idle.
  3. `read_balance`: criterion 4 holds on every plan.
  4. `link_share`: criterion 2 holds on the chosen plan. Qwen3.8 at N = 4, λ 70, A low uses at most 3 nodes. The
     predictor's plan of that cell uses two of the four: a third switched node exposes 2.270 of 22.351 ms, 10.2 %.
  5. `nic_not_max`: criterion 5 either holds or carries the named prompt-mode note, exactly in the cells of the "fires"
     column. The gate pins the ~~predictor's~~ derived cells: at S₁ 3,000, V4.1 and GLM at N ≥ 2, except three of V4.1's
     four cells at N = 2, and never Qwen3.8; at 1,000, none. The figures that follow are the predictor's; the gate holds
     each busiest NIC to the derived one within one id's change. Node 0 takes the rows of the remotes' share as the
     split sets it, where 9.8 assumed an even split: GLM at N = 2, λ 25, A low splits its ids [111, 177], and node 0's
     NIC is busy 551 ‰. In the cells that fire, the busiest NIC runs from 500 to 941 ‰ for V4.1 and from 531 to 1,007 ‰
     for GLM; Qwen3.8's stays at or below 465 ‰.
  6. `messages_o_n`: criterion 6. The busiest node posts 2 × min(k, N − 1) messages a layer, because node 0 sends the
     handoff only to the remotes that hold one of the token's slots in that layer.
  7. `never_slower`: `NodeIdle` names exactly the cells of criterion 7's column, and no chosen plan is slower than its
     N − 1 plan. The gate's cells are the column's but N = 4, λ 70, A low, where criterion 2 names the node first. It
     compares each plan with the N − 1 plan on the same link (a pair for a ring of three, three switched Sparks for
     four). The N − 1 plan on its own fabric can be faster: Qwen3.8 on four switched Sparks takes 20.997 ms at λ 25,
     A low, and on a ring of three 20.955 ms, by the switch's hop.
  - C3 of `tierengine.md` (k), `fills_balance`: the split counts each node's NVMe fills and pages no id while a pool
    has room.
- **FAIL-first, one mutant a clause:**
  - M1: `PoolOver` over `all_cards()` into node 0's pool → clause 2 red at N ≥ 2.
  - M2: plan only `nodes[..2]` (a hidden 2) → clause 2 red, because V4.1 is not whole at N = 4.
  - M3: the tier rule by prefix (today's) → clause 3 red at N ≥ 2.
  - M4: λ = 0 in the predictor → clauses 4 and 7 red, because node 3 is used for Qwen3.8.
  - M5: drop the NIC term → clause 5 red, because no note appears in its cells.
  - M6: charge a node term at N = 1 → clause 1 red.
  - M7: `ruler` fixed at 2 % instead of read from the machine → clause 7 red on a synthetic machine whose ruler is 4 %
    (Qwen3.8 at N = 2 then gains under 8 % and must name node 1 idle).

## 10. Still open

Each open item; the owner is named where one is agreed. An item that a measurement closed is in the list after it.

**P1 and the card.**
- The V4.1 DSpark draft fixture stays unpublished: no public file has its header. Only a gate that runs the draft
  refuses, after [#159](https://github.com/midagedev/bloomery/pull/159), so `gate-gpu-ds41-step` opens without it.
- Which `llc` a build names ([#54](https://github.com/midagedev/bloomery/issues/54), section 6).

**P4.**
- An arena at serving defaults. The default plans of GLM-5.3-Flash and V4.1 on one Spark have none (section 8). A
  given `BLOOMERY_NVTIER_BYTES` alone is refused there: the lowered card budget leaves the room just above the NVMe
  tier's floor, and the arena's least on GLM `UD-Q4_K_XL` is 23,867,015,168 B. (GLM with `BLOOMERY_NVTIER_BYTES=24G`,
  on this change merged with #120's head `905e7ac5` before either reached main: a room of 120,021,311,488 B against
  floors of 120,017,099,844 B and 120,013,539,120 B in its refusals.) With a lower `BLOOMERY_CARD_BUDGET` given as well
  (`BLOOMERY_CARD_BUDGET=70G BLOOMERY_NVTIER_BYTES=23G`), GLM plans a 38-slot arena (24,696,061,952 B), loads and
  answers (on `905e7ac5`; under a given card budget this change plans as before).
- The card leg: the contributor (answer 19).
- `probe_nvread`'s per-part drop (`evict_rows`) leaves some pages of a part resident on ext4, so a run redraws batches.
  The cause is not proven; one guess is large folios that straddle two experts
  ([#71](https://github.com/midagedev/bloomery/issues/71#issuecomment-6126758679)).

**P5.**
- The `[derived]` bands and criteria of 9.7 and 9.8 still use their first inputs. `nodes_scale`'s derivation script, in
  the placement half, recomputes them (the placement half as proposed in [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134316241), accepted in
  answer 20). `tools/flow/nodes_scale.py` (9.9) recomputes 9.8's "where each fires" table and 9.7's (a) rows at the
  gate's inputs. 9.7's one-node and NVMe rows, (b), (c) and N requests in flight still use their first inputs: they need
  ρ, the NVMe rate, a pipeline model, the union R(2) and a batched step, none of which is an input of the script.
- A fill term in the node plans for rows read from the NVMe tier, so that they can plan the V4.1 file (9.7, 9.9
  Terms).
- Where `nodes_scale` ([#95](https://github.com/midagedev/bloomery/pull/95)) holds a clause of 9.9 differently from its
  words, for the maintainers to accept or change: it pins the cells `tools/flow/nodes_scale.py` derives, which differ from
  9.8's table as first written in criterion 5 (clause 5); clause 1 is byte for byte only where node 0 holds every id,
  since today's plan of one Spark pages V4.1 and GLM under a card budget lowered to the NVMe tier's floor
  (`plan_in_pool`, section 8) and the node plans take node 0's terms before that split; clause 7 compares each plan
  with the N − 1 plan on the same link, which the planner can choose, not on the N − 1 plan's own fabric.
- The placement half's engine side: the contributor (answer 20). It is a remote node's census over the control channel
  (9.2), the API that builds the card plans node 0's and each remote's budget come from, the router's counts in the
  split (9.2), and decode at n = N in the predictor (9.8).
- GLM's hit rates are assumptions; V4.1's come from a replay (9.7).
- S₁ has a first reading on one Spark (9.7), at `v0.3.0` serving defaults with the MTP draft on.
- The switch hop: no switched Spark was measured (9.6).
- Whether to take tensor parallel first instead of (a) is the maintainers' decision (9.4).

**Closed.**
- ~~`glm5next/corpus-prose.ids`, `engram/corpus-prose.ids` and `fx_ref_glm5next_vis_taps` unpublished~~ on the refset
  dataset since 2026-10-14 ([#53](https://github.com/midagedev/bloomery/issues/53#issuecomment-6168041480), section 1).
- ~~Which gate carries the first bit-gate pass on cc 12.1~~ (asked in [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)).
  The fixture-tier gates in section 6 carry it. They prove bands, a tie-band argmax and exact routing ids, not bits
  against cc 8.6. The logits-row diff in [#53](https://github.com/midagedev/bloomery/issues/53#issuecomment-6159465373)
  against a 3090 run closes the bits question.
- ~~LLVM 21 and `sm_121` PTX~~ ([#42](https://github.com/midagedev/bloomery/issues/42), section 6).
- ~~Per-arch `ptx-shapes.tsv` rows, so `gate-ptx-spill` can read `sm_121` PTX~~: the cc 12.1 tables (section 6).
- ~~the host-tier protocol's correctness on Arm~~ (its spin sites, Appendix A; [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6118769122)).
- ~~the card's read rate over pageable memory~~ (section 8, [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921)).
- ~~the NVMe read rate on a Spark~~ ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921), [#71](https://github.com/midagedev/bloomery/issues/71#issuecomment-6126758679)); ~~the Spark drive's NVMe rate~~ (section 8).
- ~~the latency of a direct link~~ ([#44](https://github.com/midagedev/bloomery/pull/44)); ~~a latency for a direct link~~ (9.1).
- ~~`cuStreamWaitValue32` on a page the NIC wrote~~ ([#44](https://github.com/midagedev/bloomery/pull/44)). ~~Whether
  `cuStreamWaitValue32` on a `cudaHostAlloc` page written by the NIC behaves on GB10 is `[unverified]`.~~ It does: the
  probe's stream waits on such a page with no stale line ([#44](https://github.com/midagedev/bloomery/pull/44)).
- ~~V4.1's hit rates ρ₁ and ρ₂ are assumptions.~~ A replay of a route trace recorded on a Spark gives two rows (9.7,
  [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124845802)).
- ~~GLM's A band (4.7–8.2 GB a token) is wide: the file minus routed leaves about 3.5 GB that the tensor shapes do not
  explain.~~ A header read of the `UD-Q4_K_XL` file's non-routed tensors gives A 7.36 GB and Sh 1.60 GB a token (9.7,
  [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6139352984)).
- ~~BW~~ (section 5): 208.8 GB/s on `b289130a`, 177.5 GB/s on `04d0c0ae`.
- ~~`gate-engram`'s eviction clause on a Spark~~ (answer 18). At `86022f32` an eviction drops each run of adjacent
  rows in one call (`adjacent_runs`, `crates/engram/src/lib.rs:567`; `SplitTable::evict_rows`,
  `crates/engram/src/rows.rs:196`), and the clause (`crates/engram/tests/engram.rs:706`) checks every page of the named
  experts after each eviction.
- ~~Small-message (16 to 64 KB) bandwidth in protocol mode, strict ordering and one message in flight~~ (the
  contributor, with the protocol half, answer 20). Across a direct cable between two Sparks, the link's one-way time
  is 1.52 to 1.80 µs over `ib_write_lat`'s and 0.66 to 1.81 µs over the floor, with 0 stale lines (the table).
  The floor is one message's serialization on one rail plus the one-way trip without it `[derived]`:
  floor = bytes × 8 / R + (t_typical − bytes × 8 / R₁₂₈). R is 9.1's 109.28 or 92.57 Gb/s, t_typical is
  `ib_write_lat`'s one-way time and R₁₂₈ the `ib_write_bw -t 128` rate, each the mean of three runs in the same
  session as the link's reading. The reading: two GB10s, one RoCE v2 port each, the link's RDMA ends on its pinned
  pages with strict ordering, and `link_probe --size` writing one message at a time (one-way time = half the round
  trip, p50 of 100,000 messages, the mean of three rounds).

  | Size | t_typical | `link_probe` one-way | Over t_typical | R₁₂₈ | Floor | Over the floor |
  |---|---|---|---|---|---|---|
  | 16 KB | 4.49 µs | 6.08 µs | 1.59 µs | 108.35 Gb/s | 4.48 to 4.70 µs | 1.38 to 1.60 µs |
  | 32 KB | 5.97 µs | 7.77 µs | 1.80 µs | 108.96 Gb/s | 5.96 to 6.40 µs | 1.37 to 1.81 µs |
  | 64 KB | 8.76 µs | 10.29 µs | 1.52 µs | 109.15 Gb/s | 8.76 to 9.62 µs | 0.66 to 1.53 µs |

  On one Spark, with both RDMA ends on one RoCE port (the NIC's own loopback), the same probe read 1.50, 1.59 and
  1.69 µs over that session's t_typical. At the three sizes checked, the cable's excess falls inside the band the
  one-port reading predicted for it (1.2 to 2.0 µs).

## 11. Maintainers' decisions

The maintainers' answers on #13, numbered for citation. Answers 1 to 14 keep their wording, so a section number inside
one is this page's old number; the italic line under an answer gives its state at this revision and the current section.
Answers 15 to 21 are an English rendering of the maintainers' Korean comments of 2026-10-13; the linked comment is the
authority. The maintainers let the contributor's measurements override these leanings. Answers from the same comment on
`tierengine.md` (h), S7's allocation rule and `sparkdec.md` §4.4 are recorded in those documents. The first nine
answers came on 2026-10-10.

1. **`qdot` off x86:** refuse (R) in the P0 pull request, so the crate compiles and whole-on-card models run; scalar
   mirrors (S) in P1, because the fixture tier offloads to the host tier. NEON only when a host-tier gate's wall asks for it.
   *Now: R landed in [#14](https://github.com/midagedev/bloomery/pull/14) (`d95c7ec9`), S in [#15](https://github.com/midagedev/bloomery/pull/15) (`021dab61`).*
2. **Fixture identity:** `/models/fixtures` stays the one legal root for now; on another machine, symlink it. A
   root-relative check comes when a contributor needs it, as its own change.
   *Now: holds; section 4 says what it means on a Spark.*
3. **Publishing:** the maintainers publish the reference sets and the fixture regeneration inputs per family on Hugging
   Face, each with its source model's licence file. A family whose source licence does not allow redistribution of its
   `tokenizer.*` keys is not published. The reference sets are up:
   <https://huggingface.co/datasets/midagedev/bloomery-refsets> (public, 17 sets, 6.48 GB, with each base model's
   licence).
   *Now: as above; [`local-check.md`](local-check.md) section 4 says how a contributor fetches the fixtures and sets.*
4. **The contributor gate list:** the maintainers own it, as one derived command (`local-check.md`): `just affected`
   minus real-file and real-only gates, `gate-ptx-spill` and the timing runners (a fixture-only gate stays: its tier is the
   fixture tier).
   *Now: no command prints the list yet. The P1 list was built by this rule on `697790c4` ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)); [`local-check.md`](local-check.md) section 6. `gate-ptx-spill` is on the list where its tables pin the card (section 6).*
5. **Arm evidence:** a contributor's recorded run on the Spark (its own table, with the machine named) is the Arm
   evidence; the maintainers' x86 batch is the gate for main.
   *Now: the Spark's tables are on [#13](https://github.com/midagedev/bloomery/issues/13), each with the machine named.*
6. **P2 golden:** yes. A published Clef-Flash tape, byte for byte, is the P2 criterion once it is published (answer 14);
   until then P2's criterion is "answers, plan clean, `PoolOver` holds".
   *Now: P2 is done (section 1); the Spark run's rows are the Spark's P2 tape ([#16](https://github.com/midagedev/bloomery/pull/16#issuecomment-6103154837)).*
7. **Toolkit:** CUDA 13.3 for aarch64 (sbsa) is the tested set, not DGX OS's 13.0. `sm_86` PTX through the JIT until P3;
   `sm_121` PTX is a P3 item (`BLOOMERY_ARCH=sm_121`, section 3.2).
   *Now: `sm_121` PTX was checked on GB10 ([#42](https://github.com/midagedev/bloomery/issues/42)), and both arches have
   cc 12.1 `ptx-shapes` tables; section 6.*
8. **64 GB Spark:** best effort, for the models that fit; not a target.
   *Now: unchanged.*
9. **An aarch64 release tarball:** P3 or later.
   *Now: unchanged.*

Added 2026-10-11:

10. **The local runner mode** ([`local-check.md`](local-check.md) open question 3): in `tools/box.sh` as
    `BLOOMERY_BOX_LOCAL=1`, not a separate script. The command string, the guard and the env exports have one owner;
    a second script would copy them. The contributor builds it, as offered on #13; the maintainers review it.
    *Now: built in [#38](https://github.com/midagedev/bloomery/pull/38); [`local-check.md`](local-check.md) section 3.*
11. **P2's shape:** the cut of M4 (`unified: Option<UnifiedPool>`) and its own `PoolOver`, as #16 builds them. The full
    `MachineSpec` waits for a reader.
    *Now: landed in `7507942c`; section 7.*
12. **Above the pool:** the card computes the NVMe tier's experts from their slots, through the unified tier (section 7).
    The host leg is not the Spark's path for them.
    *Now: section 8. Since S7 the tier's arena on a Spark is write-combined `cuMemHostAlloc` memory by default ([maintainers](https://github.com/midagedev/bloomery/issues/13#issuecomment-6104493159)).*
13. **Several Sparks:** one design for N nodes, run at N = 2 first; no code may assume N = 2 (section 8).
    *Now: section 9.*
14. **The P2 tape, where:** the tape's `Q4_K_M` file and the rows `e2e.py` reads are not public yet
    ([#16](https://github.com/midagedev/bloomery/pull/16)). The maintainers publish the Clef-Flash P2 tape and its
    reference rows in the refsets dataset (answer 3). Until then P2's criterion is "answers, plan clean, `PoolOver`
    holds".
    *Now: as answer 6.*

Added 2026-10-13 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134411281), [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134437490), [#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6134443223)):

15. **Numbers a measurement proved wrong:** strike them through, do not delete them, and put the corrected value and the
    measurement's comment beside them (`AGENTS.md`). The rows the maintainers said on #41 and #42 they would fix (the
    card section, the open list and the P/E row) may be fixed in the same rewrite, if the contributor wants.
    *Now: done in this revision (sections 6 and 10, Appendix A).*
16. **The hit-rate rows:** the maintainers lean to showing both rows (the ranked fixed set and LRU); whoever read the
    trace decides.
    *Now: both rows (9.7).*
17. **`cnode_probe` (`tierengine.md` §6.2 row 6):** a direct run is enough, so `time-gate.sh` needs no local opt-in for
    it; name the SM clock and the commit.
    *Now: the GB10 `c_node` in 9.7 is such a run.*
18. **`gate-engram` and the fixture gaps:** for the eviction clause the maintainers lean to (b), adjacent rows in one
    call; whoever measured the cause decides. The fixture and public set gaps and `--place gate` are the maintainers'; a
    contributor may take any of them.
    *Now: the eviction clause is done as (b), with each run of adjacent rows dropped in one call (section 10, Closed).
    The contributor takes `--place gate` and the two qwen4exp gates' tier check; the rest is in section 10.*
19. **S7, `devmap` and the card leg:** the contributor's, `devmap`'s clause included. A design that departs from (d) and
    (h) goes ahead on its grounds; the contributor decides the load shape and the slice alignment from the
    measurements. Verification may run on any card, and the maintainers take a result checked on the contributor's card
    (GB10, RTX 5090 or others) as it is. The A6000 is the maintainers' card for their own timing; the public numbers
    come from the 3090 alone and the 3090 with the A6000.
    *Now: S7 is in, and the card leg is open (section 8). Since `8fb237f2` any card's number with its evidence can be
    published, in a table of its own for that card (`AGENTS.md`).*
20. **P5:** split as proposed. The placement half need not wait for S10 and goes up when ready; where it overlaps the
    maintainers' train, they rebase. The protocol half starts when the contributor judges P1's protocol part closed from
    its results.
    *Now: the contributor judged it closed on `b289130a`, started with the `Link` seam and added the counter relay that
    carries the crossing counters, on the plan the maintainer agreed to in [#127](https://github.com/midagedev/bloomery/issues/127#issuecomment-6155272828); section 9.*
21. **The serving-default load refusal of GLM and V4.1:** the contributor's, as a pull request after one review.
    *Now: done in [#100](https://github.com/midagedev/bloomery/pull/100): a plan the NVMe tier's floor refuses on one pool is planned again under a lower card
    budget, and both models load at serving defaults; section 8. [#93](https://github.com/midagedev/bloomery/pull/93)
    (`7a7a282b`) is a separate fix: a GLM split NextN plan's host headroom took the draft's routed experts out twice.*

## Appendix A. The P0 site map

P0 is done ([#14](https://github.com/midagedev/bloomery/pull/14), `d95c7ec9`, design R). This appendix keeps the map
the work was cut from. Its lines are at `6b630e5e` unless a line names another commit.

Two earlier documents sized this work: [machine-axes.md](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/machine-axes.md)
(the `machineaxes` memo, 2026-09-27, whose round names `sparkprep` S1 to S4, `archkey` and `isarefuse` the table below
uses, and whose items M1 and M4 sections 6 and 7.1 cite) and [rebuild.md](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md) (the "machineaxes"
result bullets, lines 197 to 200). Their line numbers are stale; the citations here are re-read.

| Round (memo name) | What | In the tree today |
|---|---|---|
| `sparkprep` S1: cfg guards on the x86 bodies | `qdot`, `deepseek2/attn`, one bin | done in [#14](https://github.com/midagedev/bloomery/pull/14) (P0): every AVX2 item under `cfg(target_arch = "x86_64")`, design R off it |
| `sparkprep` S2: per-triple `extra-rustflags` | fork patch | done: the fork's `87bc9518` reads `[target.<triple>] extra-rustflags`; `.cargo/cuda-oxide.toml` and `.cargo/config.toml` carry `znver3` for x86_64 and `native` for aarch64, and `tools/check-rustflags.sh` holds both triples |
| `sparkprep` S3 / `archkey`: one owner of `--arch` | 139 recipe lines | done: the justfile's `arch` variable reads `BLOOMERY_ARCH` (default `sm_86`; unset, every recipe expands as before), the tools that run on the box read it through `tools/arch.sh`, `tools/box.sh` forwards it, and `tools/check-recipes.sh` fails on an arch literal. `just narrow` and `release-build` refuse another arch by name; `ptx-scan` also takes `sm_120` and `sm_121`, and `gate-ptx-spill` reads a table per arch and card capability (section 6) |
| `sparkprep` S4: tools for N cards | `cards.sh`, `timing-card.sh`, `gpu-gate.sh` | ~~not done: `tools/ref/cards.sh:31-35` names two cards~~ partly: local mode ([#38](https://github.com/midagedev/bloomery/pull/38)) and the single-card timing mode ([#39](https://github.com/midagedev/bloomery/pull/39)) run on one local card; `tools/ref/cards.sh:33-34` still names the box's two |
| `isarefuse`: CPU check at `main` | x86-64-v3 refusal | partly: `crates/serve/src/preflight.rs` and `crates/gpu-gates/src/bin/bloomery_serve.rs:70-71` (`cpu_gate`, x86-only); no other bin calls it |

Order inside P0, as it was done: gate the import and the 39 functions first (the crate then compiles with every call
site still behind `has_features`), then the predicate split of answer 1, then `attn.rs`, then `ops.rs`. Each step can
be checked by the cross check before any Spark run.

`python3 tools/recipes.py pure-crates` (run on the Mac) already classifies the crates. Twelve are pure (`bloomery-decision`,
`-hf`, `-jinja`, `-levers`, `-models`, `-placement`, `-refset`, `-runtime`, `-sampler`, `-serve`, `-tokenizer`, `-vision`):
no x86 code, no device root. Three reasons in its output block aarch64 **Linux**: `x86_64 intrinsics`, `#[target_feature]`,
`is_x86_feature_detected!`. The other reasons (Linux `/proc` paths, `libc` items such as `sched_setaffinity`,
`posix_fadvise`, `RUSAGE_THREAD`) are all POSIX on Linux and are not blockers on DGX OS `[unverified until a compile]`. No
crate named `crates/gpu*` contains an x86 intrinsic (`grep -rE 'std::arch|core::arch|target_feature' crates/gpu*` finds only a
tensor-name strings (`…dflash_kv_input_target_features…`) and a comment, `crates/gpu-gates/src/ik_q8_2.rs:51`).

Sizes are S ≤ a day, M a few days, L a week or more of work for someone who knows the code.

| Site | What is there | What a portable fallback needs | Size |
|---|---|---|---|
| `crates/qdot/src/lib.rs:44` | `use std::arch::x86_64::*;` unconditional: the whole crate, and with it every crate that depends on `bloomery-qdot` (`model`, `gpu-gates`; `gpu` through `model`), fails to compile. 39 `#[target_feature]` fns, 16 `is_x86_feature_detected!` sites (`:204-206`, `:280`, `:2270`, …) | `#[cfg(target_arch = "x86_64")]` on each AVX2 fn and on the import. The pattern exists in the same file: `dot_f32` and `sum_sq_f64` are already gated (`:7657-7714`), as are `crates/gguf/tests/f16.rs:23,70,102-104`, `crates/serve/src/preflight.rs:127,188`, `bloomery_serve.rs:70`. Per-function `cfg`, not a module move: `AGENTS.md` bans splitting a `#[target_feature]` body into helpers (10–13 % loss) | M (mechanical, 7.9 k lines) |
| `crates/qdot/src/lib.rs:186-228` (`supports`, `fuses`, `has_features`) | "The weight type has a kernel **and** the CPU has the ISA" are one predicate. The matmul dispatch sends everything that fails it to an f32 dequant path (`:194`, doc "anything else dequantizes the row to f32 first") | Two designs (answer 1): **R** `supports` false off x86 plus a named refusal of host-tier types (the memo's S1); **S** `supports` true and route to the scalar mirrors, which are bit-identical to the AVX2 kernels (`:42`: "Each AVX2 kernel has a bit-identical scalar mirror"). `tile_kind` (`:527`), `has_lanes` (`:884`) and the `dot_row` match (`:363`) must then ask a separate "SIMD present" predicate | S–M |
| `crates/model/src/arch/deepseek2/attn.rs:757-940, 1603-1700, 1811-2320` | 7 `#[target_feature]` fns, helpers `v_expf8` (`:760`), `lane_tree8` (`:818`), `lane_tree8x8` (`:836`); `kq_dot_simd` asserts the ISA and **panics** without it by design (`:939-944`, "no quiet fall back"); tests at `:3116-3140` | cfg-gate the kernels; `flash_simd` (`:1067-1077`) returns false off x86, which selects the scalar flash path that `BLOOMERY_FLASH_SIMD=0` already selects (`:1065-1070`); keep the named panic in `kq_dot_simd` | M |
| `crates/model/src/ops.rs:5529` | bare `std::arch::x86_64::_mm_prefetch` in `warm_touch` | no-op or the `paced` read off x86 (the aarch64 prefetch intrinsic is unstable `[unverified]`) | S |
| `crates/model/src/bin/markov-accept.rs:137-165`, `crates/model/tests/attn.rs` (6 detects), `crates/model/tests/moe.rs` (2 refs) | x86 kernels in a bin and in tests | cfg or a named skip | S |
| `.cargo/config.toml`, `.cargo/cuda-oxide.toml` | per-triple CPU flags: `znver3` (with `-sse4a`) for x86_64, `native` for aarch64, the same list in both files | none: done in S2. The flat `znver3` warned on every crate and crashed rustc (SIGSEGV) on some on aarch64 `[#13]`; `tools/check-rustflags.sh` refuses an x86 name in the aarch64 lists | done |
| `crates/threads/src/lib.rs:786-850` | pool topology from sysfs (`thread_siblings_list`, `cache/index3/*`); with no L3 file it falls back to one group (`:798`, `:806-819`) | none to compile. Default thread count is the primary-core count (`:422`: 20 on GB10) with equal chunks (`:711`), ~~so the 10 E-cores gate every barrier~~ but on the default Spark path no decode or prefill step dispatches to the pool, which runs only the load's upload ([#41](https://github.com/midagedev/bloomery/issues/41)); the pool's two groups follow the two L3 clusters (section 3), but the P/E split inside a group is not modeled | none for P0–P2; ~~M in P3~~ none on the default path ([#41](https://github.com/midagedev/bloomery/issues/41)) |
| `crates/placement/src/placement/workstation.rs:138` (at `b289130a`), `paged_drop.rs:181` | `ROW_PAGE` and `PAGE` are 4096, commented "on x86_64" | the DGX OS base page size is 4096 `[#13]`, so the constants hold; a `sysconf` read that refuses another size by name is still the right guard | S |
| `tools/mac-check.sh:69` (`TARGET=x86_64-unknown-linux-gnu`, also `:178,192,321-323`, `BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu` `:45,330`) | the Mac's cross check | take the triple from `BLOOMERY_CHECK_TARGET`: P0 is then provable without a Spark with `cargo check --target aarch64-unknown-linux-gnu` over the host chain (`-p bloomery-qdot -p bloomery-model -p bloomery-threads …`, no bindgen needed). Needs `rustup target add --toolchain nightly-2026-08-28 aarch64-unknown-linux-gnu`, which the Mac's toolchain lacks today | S |

Third-party dependencies are portable (`Cargo.lock` has `cpufeatures`, `half`, `sha2`, `libc`, `memmap2`; none is x86-only).
`std::ffi::c_char` is used for the driver's name buffers (`crates/gpu/src/lib.rs:1749,1791`), so the aarch64 `u8` `c_char`
is not a trap in the sites grepped.

**Memory ordering.** NVIDIA's porting guide warns that "some reorderings cannot happen on x86_64 but can happen on ARM"
(`.../porting/memorder.html`). The host tier's card/host protocol spins on host-mapped words (`crates/gpu/src/host/mod.rs:466`,
`step.rs:1287`, `tier.rs:1476`, `lane.rs:513`, `xstream.rs:431`, `swap.rs:3761`: 6 `spin_loop` sites) and
`crates/gpu/src/host/*.rs` holds 29 `Ordering::Relaxed` loads. ~~Not audited; it is a P1 risk for the host-tier gates, and
irrelevant to a whole-on-card run that never starts a host service.~~ The audit found no site in the host tier that is
correct only under x86's ordering, and no code change is needed
([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6118769122), which read main `23ee2100`). It counts
121 `Relaxed` tokens under `crates/gpu/src/host/` (loads, stores and read-modify-writes, `nvtier/reader.rs` included;
the 29 above are loads only, at `6b630e5e`), and they are stat counters, same-thread data or books written under a lock. The one
point the code could not settle, whether the GB10 card sees a CPU Release as ordering the plain stores before it, was
probed on one Spark: with the payload pushed out of the cache, a plain `str` flag let the card read stale lines, and a
`stlr` or `ldaddl` publish did not.
