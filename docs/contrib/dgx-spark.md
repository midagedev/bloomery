# Porting bloomery to the NVIDIA DGX Spark

A map of what stands between this tree and "builds and passes the fixture-tier gates on a DGX Spark", written for a
person who owns one. Every claim about the code cites `path:line` as of main `6b630e5e` (sections 1–3.5; a line re-read
on 2026-10-11 is at `61039d6e`) or `61039d6e` (sections 3.6 and 7; section 8's code citations were re-read at `cce176c5`);
a thing that was not seen is marked `[unverified]`, a number that is computed here is marked `[derived]`. The maintainers
ran nothing on a Spark. A contributor's readings on two DGX Sparks are in section 1 and marked `[#13]`
([issue #13](https://github.com/midagedev/bloomery/issues/13)).

Two earlier documents already sized this work: [machine-axes.md](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/machine-axes.md) (the
`machineaxes` memo, 2026-09-27; its round names `sparkprep` S1–S4, `archkey`, `isarefuse` and "M1–M3" are used below) and
[rebuild.md](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md) (the "machineaxes" result bullets, lines 197–200). Their line numbers are stale; the
citations here are re-read. Related: [`new-model.md`](new-model.md) (the other contribution track),
[`local-check.md`](local-check.md) (the off-box check design; this file keeps only what is Spark-specific).

**Where it stands (2026-10-11).** P0 is in main ([#14](https://github.com/midagedev/bloomery/pull/14)). P1's `qdot`
part is [#15](https://github.com/midagedev/bloomery/pull/15), reviewed. P2's pool invariant is
[#16](https://github.com/midagedev/bloomery/pull/16), in review. The maintainers are finishing the unified tier, which
section 7 needs. Section 8 is the design for several Sparks: sourced link facts and derived bands, nothing run on two
Sparks. The Spark-prep rounds of the earlier memo:

| Round (memo name) | What | In the tree today |
|---|---|---|
| `sparkprep` S1: cfg guards on the x86 bodies | `qdot`, `deepseek2/attn`, one bin | done in [#14](https://github.com/midagedev/bloomery/pull/14) (P0): every AVX2 item under `cfg(target_arch = "x86_64")`, design R off it |
| `sparkprep` S2: per-triple `extra-rustflags` | fork patch | done: the fork's `87bc9518` reads `[target.<triple>] extra-rustflags`; `.cargo/cuda-oxide.toml` and `.cargo/config.toml` carry `znver3` for x86_64 and `native` for aarch64, and `tools/check-rustflags.sh` holds both triples |
| `sparkprep` S3 / `archkey`: one owner of `--arch` | 139 recipe lines | done: the justfile's `arch` variable reads `BLOOMERY_ARCH` (default `sm_86`; unset, every recipe expands as before), the tools that run on the box read it through `tools/arch.sh`, `tools/box.sh` forwards it, and `tools/check-recipes.sh` fails on an arch literal. `gate-ptx-spill`, `ptx-scan`, `just narrow` and `release-build` refuse another arch by name; the per-arch pin rows are P3 |
| `sparkprep` S4: tools for N cards | `cards.sh`, `timing-card.sh`, `gpu-gate.sh` | not done: `tools/ref/cards.sh:31-35` names two cards |
| `isarefuse`: CPU check at `main` | x86-64-v3 refusal | partly: `crates/serve/src/preflight.rs` and `crates/gpu-gates/src/bin/bloomery_serve.rs:70-71` (`cpu_gate`, x86-only); no other bin calls it |

## 1. Facts

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
| NEON / SVE2 / dot-product / i8mm | not stated by NVIDIA on any page read. The SIMD page (`.../porting/simd.html`) speaks of SVE and SVE2 in general only | `[unverified]`; first action on a Spark: `grep -m1 Features /proc/cpuinfo` |
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
| Build | `cargo oxide doctor` passes; the flat `target-cpu=znver3` makes rustc crash (SIGSEGV in LLVM) on some crates, so a Spark owner sets `target-cpu=native` in the local `.cargo/cuda-oxide.toml` until S2 |
| First run | Qwen3.6-35B-A3B `Q4_K_M` loads whole on the card and answers, after `drop_caches`: before it the plan read 3,762,335,744 B free, put 10,102 experts on the host tier, and the host union refused every `Q4_K` stack by name (section 3.3's invariant in a P0 run); the sm_86 PTX JIT-compiles for cc 12.1 on driver 595.84 |

## 2. The cost model on paper

The question for the port is what a Spark is for. The decomposition below uses only repo numbers and the facts above.

**Anchor [measured, not re-measured since 0.2.3].** Qwen3.6-35B-A3B `Q4_K_M`, whole model on one RTX 3090 at its 250 W
cap: 193.9 tok/s, one request's rate (`README.md:111` and footnote 1), which is 5.16 ms a token `[derived]`.

**Terms of one decode token, whole model on the card:**

| Term | 3090 box | Spark | How it is obtained |
|---|---|---|---|
| Weight bytes a token reads | 1.8 GB | 1.8 GB | 3 B active parameters (the "A3B") × 4.85 bits (21.2 GB file, `docs/models.md:33`, over 35 B parameters) `[derived]` |
| Byte time at the memory | 2.6 ms at ~700 GB/s (`AGENTS.md`, "a grid near ~700 GB/s") | 6.7 ms at the 273 GB/s peak; 8.3 ms at 80 % of it `[assumed 80 %]` | bytes / bandwidth |
| Rest of the step (launch chain, attention, norms) | 5.16 − 2.6 = 2.6 ms `[derived]` | 2.6 ms if it is latency, 4.4 ms if it scales with SMs (82 / 48 = 1.7; the 3090's 82 SMs are `[unverified]` here) | — |
| H2D / PCIe | 0 | no PCIe leg for weights | the whole-card seat has no host tier |
| NVMe | 0 after load | 0 after load; the load reads the file once | — |

**Prediction [derived]:** a step of 8.3 ms (bytes hide the rest) to 12.7 ms (bytes plus an SM-scaled rest), so 79–120 tok/s,
point 92 tok/s (bytes plus latency, 10.9 ms). That is 0.4–0.6 of the 3090's rate, and the byte term is 2.6–3.2 times
longer than on the box. The `machineaxes` round reached 2.47× for the A6000 (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md#L200).

**Resource timeline per decode token (whole on card, Spark).** Host CPU: a graph launch and the sampler, off the critical
path (no host tier), cost `[unmeasured]`. Card SMs: the 2.6–4.4 ms rest, overlapped with the byte stream. Card DRAM and
host DRAM are **one resource**, the 273 GB/s pool: the 6.7–8.3 ms byte term. PCIe: 0. NVMe: 0. The wall is that one
resource's time (or at most bytes plus the latency part), not the sum of five. Consequences:

- Host/card asynchrony (overlapping CPU expert work under card work) is worth 0 here: there is no host expert work to hide
  and the CPU's DRAM traffic competes with the card's.
- Once the byte term is at the pool's sustained rate, a kernel change that removes instructions predicts 0
  (`AGENTS.md`, "instruction count on a kernel near ~700 GB/s: model says 0"). The levers left are fewer bytes (quant, KV
  size), launch count when the latency part is the larger one, and SM occupancy only if the rest is compute.
- A Spark's value is its 128 GB, not its bandwidth: V4.1 `Q3_K_M` (347.3 GB) and GLM-5.3-Flash `UD-Q4_K_XL` (199.7 GB)
  (`docs/BUILD.md`, model table) do not fit the pool and would need the NVMe tier; Qwen3.8 `UD-Q4_K_XL` (111.3 GB) fits
  with 13.9 GB of the idle `MemAvailable` (116.6 GiB = 125.2 GB, section 1) left for KV and runtime `[derived from #13]`.

**The one term a run must settle, with its expected value:** the sustained read rate of our decode gemv kernels on the
LPDDR5x with the CPU idle. Expected 200–235 GB/s (73–86 % of 273). A plain read kernel measured 233 GB/s on `cuMemAlloc` memory (0.85 of 273)
and 235 GB/s on write-combined `cuMemHostAlloc` pages ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6102768921));
our gemv kernels' own rate is still unread. Decision rule: at or
above 200 the decode flow is byte-bound and P3 kernel work predicts 0 beyond byte-saving levers; below 160, look at page
size, TLB reach and clocks (`nvidia-smi` clock-event reasons) before any kernel work. Everything else in this file is
settled by reading.

## 3. What stands between the tree and a passing fixture tier

### 3.1 The host on aarch64

`python3 tools/recipes.py pure-crates` (run on the Mac) already classifies the crates. Twelve are pure (`bloomery-decision`,
`-hf`, `-jinja`, `-levers`, `-models`, `-placement`, `-refset`, `-runtime`, `-sampler`, `-serve`, `-tokenizer`, `-vision`):
no x86 code, no device root. Three reasons in its output block aarch64 **Linux**: `x86_64 intrinsics`, `#[target_feature]`,
`is_x86_feature_detected!`. The other reasons (Linux `/proc` paths, `libc` items such as `sched_setaffinity`,
`posix_fadvise`, `RUSAGE_THREAD`) are all POSIX on Linux and are not blockers on DGX OS `[unverified until a compile]`. No
crate named `crates/gpu*` contains an x86 intrinsic (`grep -rE 'std::arch|core::arch|target_feature' crates/gpu*` finds only a
tensor-name strings (`…dflash_kv_input_target_features…`) and a comment, `crates/gpu-gates/src/ik_q8_2.rs:51`).

| Site | What is there | What a portable fallback needs | Size |
|---|---|---|---|
| `crates/qdot/src/lib.rs:44` | `use std::arch::x86_64::*;` unconditional: the whole crate, and with it every crate that depends on `bloomery-qdot` (`model`, `gpu-gates`; `gpu` through `model`), fails to compile. 39 `#[target_feature]` fns, 16 `is_x86_feature_detected!` sites (`:204-206`, `:280`, `:2270`, …) | `#[cfg(target_arch = "x86_64")]` on each AVX2 fn and on the import. The pattern exists in the same file: `dot_f32` and `sum_sq_f64` are already gated (`:7657-7714`), as are `crates/gguf/tests/f16.rs:23,70,102-104`, `crates/serve/src/preflight.rs:127,188`, `bloomery_serve.rs:70`. Per-function `cfg`, not a module move: `AGENTS.md` bans splitting a `#[target_feature]` body into helpers (10–13 % loss) | M (mechanical, 7.9 k lines) |
| `crates/qdot/src/lib.rs:186-228` (`supports`, `fuses`, `has_features`) | "The weight type has a kernel **and** the CPU has the ISA" are one predicate. The matmul dispatch sends everything that fails it to an f32 dequant path (`:194`, doc "anything else dequantizes the row to f32 first") | Two designs, see Q1: **R** `supports` false off x86 plus a named refusal of host-tier types (the memo's S1); **S** `supports` true and route to the scalar mirrors, which are bit-identical to the AVX2 kernels (`:42`: "Each AVX2 kernel has a bit-identical scalar mirror"). `tile_kind` (`:527`), `has_lanes` (`:884`) and the `dot_row` match (`:363`) must then ask a separate "SIMD present" predicate | S–M |
| `crates/model/src/arch/deepseek2/attn.rs:757-940, 1603-1700, 1811-2320` | 7 `#[target_feature]` fns, helpers `v_expf8` (`:760`), `lane_tree8` (`:818`), `lane_tree8x8` (`:836`); `kq_dot_simd` asserts the ISA and **panics** without it by design (`:939-944`, "no quiet fall back"); tests at `:3116-3140` | cfg-gate the kernels; `flash_simd` (`:1067-1077`) returns false off x86, which selects the scalar flash path that `BLOOMERY_FLASH_SIMD=0` already selects (`:1065-1070`); keep the named panic in `kq_dot_simd` | M |
| `crates/model/src/ops.rs:5529` | bare `std::arch::x86_64::_mm_prefetch` in `warm_touch` | no-op or the `paced` read off x86 (the aarch64 prefetch intrinsic is unstable `[unverified]`) | S |
| `crates/model/src/bin/markov-accept.rs:137-165`, `crates/model/tests/attn.rs` (6 detects), `crates/model/tests/moe.rs` (2 refs) | x86 kernels in a bin and in tests | cfg or a named skip | S |
| `.cargo/config.toml`, `.cargo/cuda-oxide.toml` | per-triple CPU flags: `znver3` (with `-sse4a`) for x86_64, `native` for aarch64, the same list in both files | none: done in S2. The flat `znver3` warned on every crate and crashed rustc (SIGSEGV) on some on aarch64 `[#13]`; `tools/check-rustflags.sh` refuses an x86 name in the aarch64 lists | done |
| `crates/threads/src/lib.rs:786-850` | pool topology from sysfs (`thread_siblings_list`, `cache/index3/*`); with no L3 file it falls back to one group (`:798`, `:806-819`) | none to compile. Default thread count is the primary-core count (`:422`: 20 on GB10) with equal chunks (`:711`), so the 10 E-cores gate every barrier; the pool's two groups follow the two L3 clusters (section 1), but the P/E split inside a group is not modeled | none for P0–P2; M in P3 |
| `crates/placement/src/placement/workstation.rs:132`, `paged_drop.rs:181` | `ROW_PAGE` and `PAGE` are 4096, commented "on x86_64" | the DGX OS base page size is 4096 `[#13]`, so the constants hold; a `sysconf` read that refuses another size by name is still the right guard | S |
| `tools/mac-check.sh:69` (`TARGET=x86_64-unknown-linux-gnu`, also `:178,192,321-323`, `BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu` `:45,330`) | the Mac's cross check | take the triple from `BLOOMERY_CHECK_TARGET`: P0 is then provable without a Spark with `cargo check --target aarch64-unknown-linux-gnu` over the host chain (`-p bloomery-qdot -p bloomery-model -p bloomery-threads …`, no bindgen needed). Needs `rustup target add --toolchain nightly-2026-08-28 aarch64-unknown-linux-gnu`, which the Mac's toolchain lacks today | S |

Third-party dependencies are portable (`Cargo.lock` has `cpufeatures`, `half`, `sha2`, `libc`, `memmap2`; none is x86-only).
`std::ffi::c_char` is used for the driver's name buffers (`crates/gpu/src/lib.rs:1749,1791`), so the aarch64 `u8` `c_char`
is not a trap in the sites grepped.

**Memory ordering.** NVIDIA's porting guide warns that "some reorderings cannot happen on x86_64 but can happen on ARM"
(`.../porting/memorder.html`). The host tier's card/host protocol spins on host-mapped words (`crates/gpu/src/host/mod.rs:466`,
`step.rs:1287`, `tier.rs:1429`, `lane.rs:513`, `xstream.rs:431`, `swap.rs:3761`: 6 `spin_loop` sites) and
`crates/gpu/src/host/*.rs` holds 29 `Ordering::Relaxed` loads. Not audited; it is a P1 risk for the host-tier gates, and
irrelevant to a whole-on-card run that never starts a host service.

### 3.2 The card

- **Targets today.** Recipes pass `--arch {{arch}}`, the justfile's variable (139 lines): `BLOOMERY_ARCH`, default `sm_86`,
  so `BLOOMERY_ARCH=sm_121 just <recipe>` passes `--arch sm_121`; whether the pinned toolchain accepts it is `[unverified]`
  (the `sm_121` bullet below). `tools/gate.sh`, `tools/lds-scan.sh` and
  `tools/sass-scan.sh` read it through `tools/arch.sh`; `tools/ptx-scan.sh` refuses any arch but `sm_86`. The kernels are SIMT code with no generation-specific
  instruction (the three `ptx_asm!` sites are `cvta`, `ld.global`, `max.f32`: `crates/gpu/src/linear/delta.rs:121-157`,
  `flash_gqa_prefill.rs:289`; `fault.rs:714` not read).
- **Documented state.** `docs/cuda-oxide.md:5`: "Everything here runs on Ampere (sm_86) only." The toolchain ledger
  (`docs/upstream/nvlabs-ledger.md`) has no `sm_121` or Blackwell row; its only Spark-related entry is row 32 (the flat
  `extra-rustflags`). So no defect of the pinned cuda-oxide on Blackwell is known, because none has been tried.
- **P0–P2 need no new target.** The binary carries `.target sm_86` PTX and the driver JIT-compiles it at load; the release
  states it (`tools/release/README.release.md:15-16`: "the archive carries sm_86 PTX, which the driver compiles for the
  card on first start"), and the loader says it for Blackwell: "A payload built for a standard pre-Blackwell target, such as
  `sm_86`, may instead be converted to PTX and JIT-compiled by the driver on Blackwell"
  (`cuda-host/src/embedded.rs:58-62` at fork rev `4da6c13`, `Cargo.toml:30`), and the first run on a Spark did it
  (section 1 `[#13]`). First JIT cost on the Spark's Arm cores is `[unmeasured]` (10–25 s measured on the box,
  `README.release.md`).
- **`sm_121` for P3.** The fork's target table has `capability: 121` (`cuda-target-spec/src/lib.rs:382,417,447`) and its
  selection tests name `sm_121a`/`sm_121f` (`cuda-oxide-codegen/src/target/tests/selection.rs:162-179`). Whether LLVM 21's
  NVPTX backend accepts `sm_121` is `[unverified]`; `cargo oxide doctor` passes on aarch64 `[#13]`. The fork's `tcgen05` path
  is datacenter-Blackwell only (`rustc-codegen-cuda/README.md:107-108`: "need `sm_100a`, which a consumer `sm_120` lacks"),
  and we use no such instruction.
- **Pins that follow the compute capability.** `tools/ref/ptx-shapes.tsv:6` pins spill and `jit_local` from "sm_86, JIT on
  the RTX 3090" and carries a `# arch: sm_86` line; `tools/recipes.py:1960-1965` says the JIT columns follow the capability;
  `tools/ptx-scan.sh` hard-codes the sm_86 occupancy column. So under `BLOOMERY_ARCH=sm_121` `gate-ptx-spill` ends with "no
  ptx-shapes rows for sm_121; the shape pins are sm_86 only" and `ptx-scan` (so `just narrow`) refuses by name, and no sm_121
  scan is read against sm_86 rows; per-arch rows are P3 (memo M1). A Spark contributor leaves them out.
- **Bit gates.** The memo concludes the bit gates have no generation dependence by structure (they compare against oracle
  files) and names two residual risks: `ptxas` contracting `.rn`-less mul-then-add differently, and the `ex2.approx`-class
  approximate instructions in about 30 places (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/rebuild.md#L198, https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/machine-axes.md#L10-L16). The proof is
  one bit-gate pass on the Spark (`gate-gpu-e2e`'s count pin and the card family gates). The repo has only ever run
  cc 8.6 (`tools/recipes.py:1960`: "both cards here are sm_86").
- **Device pick.** The real-tier gate plan opens the card by the names in `CARDS` and refuses any other card by name
  (`crates/gpu-gates/src/bin/shared/gate_card.rs:8-9, 38-47`: "none of this workstation's cards"). The fixture tier plans on
  `--place a`, "the largest visible card as the device reports itself" (`gate_card.rs:4-6`) and is not affected. The real
  tier is the maintainers' tier regardless (section 3.4).

### 3.3 Unified memory

Placement assumes a discrete card, a PCIe host tier and an NVMe tier (`crates/placement/src/placement.rs:223-229`:
`Machine { cards, tiers, host }`; there is no pool field at main, and the only `unified` under `crates/placement/src` is
the cgroup reader's "unified hierarchy", `workstation.rs:726-747`; [#16](https://github.com/midagedev/bloomery/pull/16)
adds `Machine::unified`, section 3.6, and for several Sparks `host` and `unified` move into `nodes[0]` of a per-node
`Machine`, section 8.2).

**The double count, concretely.** Three readers see one pool:
`cuDeviceTotalMem` (`crates/gpu/src/lib.rs:1789`), `cuMemGetInfo` (`:1995`, `:2302`: the card's `free_bytes`, which caps the
plan's card budget, `devices.rs:98-123`) and `/proc/meminfo` `MemAvailable` (`workstation.rs:667`, `host_room` `:858`: the
host room). A plan checks card bytes and host bytes each against its own figure, so on a Spark it can admit up to twice the
pool. NVIDIA adds that `cudaMemGetInfo` under-reads, because reclaimable page cache and swap are not counted (section 1), so
the card's figure is also too small right after the model file was read. The invariant needed is one: Σ card + host ≤ pool,
with the pool read as `MemAvailable` (plus what the load will drop), not `cuMemGetInfo` (memo M4: extend `Machine` and
`workstation.rs`, no parallel type).

**Which terms vanish and which stay:**

| Term | On the Spark |
|---|---|
| H2D promotion over PCIe (`cuMemcpyHtoDAsync`, `crates/gpu/src/host/xstream.rs:1223,1261` at `cce176c5`; the staging rings `:283,760,1614`) | vanishes as a PCIe cost; a copy from pageable memory into a `cudaMalloc` buffer is still a DRAM-to-DRAM copy at half the pool's rate (reads and writes both cross it: 21.2 GB in ~0.16 s `[derived]`), negligible beside the file read |
| Router D2H handoff and the host service protocol (`host/mod.rs:1-60`) | unused when no layer has host experts; works on `cuMemHostAlloc` pages (`host/tier.rs:561`, `xstream.rs:267`), the allocator NVIDIA recommends; correctness on Spark `[unverified]` |
| CPU expert compute | stays possible but pointless: 20 Arm cores against 48 SMs on the same bytes. It is the path the fixture tier exercises (3.4) |
| The pool's bandwidth | stays, as the single term (section 2) |
| Page cache of the model file plus the device copy | stays and doubles the footprint until dropped; `nvtier.rs:12-30` already drops pages after use (`MADV_DONTNEED`, the `release_union` path) for the NVMe tier |
| NVMe → DRAM stream for models above the pool | stays; its rate on a Spark is `[unverified]`. Today the NVMe tier lends its RAM slots to the host leg (`crates/gpu/src/host/nvtier.rs:1-9`), so the CPU computes those experts; on one pool the card can read the same slots without a copy, so the card computes them (P4, section 7) |
| KV cache bytes | stay, and they come out of the same pool |

**Smallest first placement: everything on the card, no host tier, no NVMe tier.** Two ways in, both generic already:
`--place a` is `Pick::Rank(0)`, "the largest visible card" (`crates/placement/src/placement/devices.rs:219-224`), and an
unknown card takes the census path with its own total and free bytes (`devices.rs:98-123`); the Qwen3 and Qwen3.6 seat
opens "CUDA device 0" (`crates/gpu/src/lib.rs:2124`, `docs/BUILD.md:88`). `--place gate` needs a card named 3090
(`devices.rs:221`) and is refused. (`docs/BUILD.md:77` still says `a` needs a card named A6000; that is stale.)

Models by size, in the order to try them (`docs/models.md:12,31-34`, `docs/BUILD.md` table): Clef-Flash 4.8–9.7 GB,
Qwen3-30B-A3B `Q4_K_M` 18.6 GB, Qwen3.6-35B-A3B `Q4_K_M` 21.2 GB; then Qwen3.8 `UD-Q4_K_XL` 111.3 GB and GLM-5.3-Flash
`UD-Q2_K_XL` 101.3 GiB (the size chosen "to fit a DGX Spark", https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan-ledger.md#L359) if the first `MemAvailable` reading
allows. A 64 GB Spark (section 1; announced, not yet shipping) holds only the first three.

### 3.4 Running the gates off our machine

What a contributor's host must not assume, and the smallest change for each (the design of the one off-box command is
`local-check.md`'s):

| Assumption | Where | Smallest change | Size |
|---|---|---|---|
| Work runs over ssh on host `ws`, in `~/repo/<tree>`, after an rsync | `tools/box.sh:42,43,108,225` | a `BLOOMERY_BOX=local` mode: run the recipe's command in the same tree, sourcing `$BLOOMERY_ENV` instead of `~/bloomery-env.sh` | S |
| Locks and holds under `/root` | `/root/bloomery-cpu.lock`, `-<owner>-hold` (`box.sh:10-11`, `tools/ref/lease-probe.sh`), `/root/bloomery-gate*.lock`, `-v41-load.lock`, `-batch.gpuhold` (`tools/gpu-gate.sh:48-50`); 41 `/root/` occurrences in `box.sh` 5, `gpu-gate.sh` 8, `lease-probe.sh` 4, `lease.sh` 5, `gate-batch.sh` 11, `justfile` 7, `ref-paths.sh` 1; in all 30 files under `tools/` name `/root/`, 17 of them under `tools/ref/` (the model profiles `tools/ref/models/*.sh` that `ref-paths.sh` sources, the dump scripts, the C++ harnesses) | one `BLOOMERY_LOCK_DIR` (default `/root`). Precedent: `gpu-gate.sh --test-locks DIR` already moves the three locks and the hold file for its self-test (header, line 17) | M |
| Two named cards | `tools/ref/cards.sh:31-35` matches the exact driver names `NVIDIA GeForce RTX 3090` and `NVIDIA RTX A6000`; `gpu-gate.sh:697-726` picks `3090\|a6000\|any\|both`; a Spark has neither, so both UUIDs stay empty | `BLOOMERY_GATE_CARD=local`: one lock under the lock dir, no name match, same bound and exit codes (`tools/gate.sh` stays the owner of the code). `nvidia-smi` prints `[N/A]` for GB10's memory fields `[#13]`; our census uses the driver API, `gpu/src/lib.rs:1789` | M (`gpu-gate.sh` is 1,021 lines; its self-test needs cases) |
| Data and fixture roots | `BLOOMERY_DATA` defaults to `/root/bloomery-data` (`tools/ref/ref-paths.sh:67`); fixtures to `/models/fixtures` (`ref-paths.sh:88`, `crates/refset/src/fixture.rs:23-24`) | both are env-movable already | none |
| A reference set names the fixture's path | a set's `# model` line states the first-shard path the dumper was given (`refset/src/fixture.rs:19-20`) and `AGENTS.md` makes "the first shard's full path" the identity, so a root other than `/models/fixtures` makes the check refuse the set `[unverified: the comparison itself was not read]` | either the contributor uses `/models/fixtures` literally, or the check compares root-relative (Q2) | S |
| cuda-oxide backend at a fixed path | `box.sh` exports `CUDA_OXIDE_BACKEND=$HOME/.cargo/cuda-oxide-bloomery/<rev>/librustc_codegen_cuda.so` and stops with rc 70 when it or its `source-rev.txt` is missing (`AGENTS.md`, Toolchain pins) | the contributor builds the backend at the pinned fork rev on aarch64 with LLVM 21 and records `source-rev.txt`; an aarch64 build of the backend has never been done here `[unverified]` | M, one time |

**Fixtures: what has to be published.** A fixture is "a model file much smaller than the real one that keeps every
per-layer shape" (`crates/model/src/fixture/mod.rs:1-4`) and "its bytes are a function of the seed and the source header
alone" (`:29-32`), pinned by per-type digests (`fixture/digest.rs:1-10`). `fixture generate <real first shard> <out dir>`
writes it (`crates/model/src/bin/fixture.rs:4-5`). Three families have fixtures (`crates/refset/src/fixture.rs:33-37`:
`qwen4exp`, `deepseek41`, `glm5next`); every other family is `self` in the table (`ref-paths.sh:75-79`) and its
gates are deferred by name as real-file. So "P1 fixture gates" means those three families plus the host gates whose
inputs are in process or published, not every `gate-*` recipe. A text classification of the `justfile` (not the batch
tool's): of 136 classified recipes, 44 are host-only and 60 take a card in both tiers, and 32 are real-tier only
(`real-only.sh` or an inline exit 66); several host gates (`gate-qdot`) read harness dumps under `$BLOOMERY_DATA`.

What a contributor cannot make off our box is the **reference sets**: they are `ik` node dumps of each fixture
(`dump-ref-fixture`, `justfile:609`) from the C++ harness the maintainers build (`just build-ref`). The maintainers publish
them: the reference sets are at <https://huggingface.co/datasets/midagedev/bloomery-refsets> (public, 17 sets, 6.48 GB,
with each base model's licence), and how to regenerate the fixtures goes with them. The fixture files are about 49 GB in all, measured on the box (glm5next 8 GB, qwen38 11 GB, v41 16 GB, v41-r8
7.3 GB, the `fx_` reference sets 6.4 GB; see [`local-check.md`](local-check.md)), but because the bytes are reproducible, publishing
them is optional: publish each source model's header (a few MB per
shard) and the seed, plus the sets. Whether `fixture generate` reads only the source's header is `[unverified]` (the
`gate-fixture` recipe's "Reads headers only", `justfile` near line 1013, is about the gate); whether the fill is
bit-identical on aarch64 is answered by the digest test, run once there. A licence question comes with publishing: the
generator copies the source's `tokenizer.*` and `general.*` keys (`fixture/mod.rs:12-14`), so a fixture carries the source
model's tokenizer (Q3).

**What the fixture tier exercises on a Spark.** The fixture header "names the card budget that forces the plan to offload"
(`ref-paths.sh:33-35`), so the fixture gates run the **host tier**: `qdot`, the CPU attention, the card/host protocol. A
Spark contributor therefore needs design S for `qdot` at P1 (scalar mirrors, correct bits, slow), not just design R.

### 3.5 Timing

The timing runners pin the A6000 by name: `tools/ref/timing-card.sh:36` (`TIMING_GPU=${BLOOMERY_TIMING_GPU:-$GPU_A6000}`),
and a UUID that is neither named card is refused while a named card is missing (`:92-93`). A Spark number is admissible
when a lease runner on the contributor's own machine produced it and it sits in the machine's own table, never beside ours
(`AGENTS.md`: "3090 numbers … and A6000 numbers never share a table"; the same holds for a third machine). What the runners
need:

- A single-card timing mode: the lease file under `BLOOMERY_LOCK_DIR`, no 3090/A6000 branches, a prediction card per run
  (`BLOOMERY_LEASE_CARD`, `tools/ref/card.py`). M.
- A witness for this machine: card name and power limit (`timing-card.sh:130`; `power.limit` is `[N/A]` on GB10
  `[#13]`), clock-event counters, the Xid count (`:27`, needs the kernel journal), and, because the pool is shared,
  CPU load, `MemAvailable` and page-cache state at the start and end of every arm (NVIDIA's `drop_caches` line is the
  debugging aid, section 1). `cpu-freq` reads `scaling_cur_freq`, present on GB10 (driver `cppc_cpufreq`) `[#13]`.
- Every number carries `tok/s @ n=N, depth D, card` (`AGENTS.md`) plus the machine line: GB10, memory, DGX OS, driver,
  CUDA toolkit, clocks.
- The ruler is theirs: the 0.6 % same-binary SD is our box's (`AGENTS.md`, "Know the ruler"). The first Spark run is an
  A/A of at least six rounds, and `tools/ref/card.py` then sets the round count for any claim.
- A reference engine arm, interleaved in the same lease (`AGENTS.md`, "the line to beat"), is a build the contributor makes
  for aarch64; no repo recipe does it today.

### 3.6 The pool invariant as #16 builds it

[#16](https://github.com/midagedev/bloomery/pull/16) (in review) takes M4's shape cut to what P2 reads:

- `census()` reads `CU_DEVICE_ATTRIBUTE_INTEGRATED` into `DeviceInfo::integrated`. On an integrated device the card's
  free bytes are the host's room (`MemAvailable` and the cgroup levels), not `cuMemGetInfo`. Nothing matches on a card
  name.
- `Machine` gains `unified: Option<UnifiedPool>`. On such a machine the host is the pool itself, with no OS reserve,
  because `MemAvailable` already leaves out what the OS and other processes hold.
- `Plan::violations` gains a named `PoolOver`: the cards' totals with their margins plus the host's whole need (the NVMe
  split's arena and window included) must fit the pool. `HostNeed::cards` gives the cards their share wherever the host's
  need is read.
- On a card with its own memory nothing moves: `integrated` is false, `unified` is `None`, `HostNeed::cards` is 0. The
  maintainers' review checks this claim hunk by hunk, and the x86 landing batch is the gate.
- A model above the pool is refused by name (`PoolOver`); sizing it onto the NVMe tier is P4's (section 7).

The placed fill keeps its card budget. On one pool an expert costs the same bytes on either side, so a split does not
change the sum, and moving experts to the host only moves them to the slower side (section 3.3's table). The default
placement on a Spark stays "everything on the card".

## 4. Phased plan

Sizes are S ≤ a day, M a few days, L a week or more of work for someone who knows the code. "Maintainers first" is what
must exist before a contributor can start; "contributor" is what the Spark's owner does.

| Phase | Done when | Files | Size | Who |
|---|---|---|---|---|
| **P0** compiles on aarch64 (done, [#14](https://github.com/midagedev/bloomery/pull/14)) | `cargo check` of the host chain passes on aarch64 Linux (on a Spark, or the cross check of 3.1 from any machine), then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe` on the Spark; and on x86 nothing moved: `just ptx-scan` equal and `gate-qdot`, `gate-ops` green | `crates/qdot/src/lib.rs`, `crates/model/src/arch/deepseek2/attn.rs`, `crates/model/src/ops.rs:5529`, `markov-accept.rs`, the two test files; `tools/mac-check.sh` triple; a local `.cargo/cuda-oxide.toml` | M | **Contributor** writes it. **Maintainers first**: decide Q1 (R or S) and answer Q5; **maintainers after**: run the x86 landing batch on the PR (the box is the only x86 judge) |
| **P1** fixture gates pass (`qdot` design S: [#15](https://github.com/midagedev/bloomery/pull/15)) | the host gates whose inputs are in process or published, and the three fixture families' gates, are green or each red is a named, Arm-only clause; a first bit-gate pass on cc 12.1 | `crates/qdot/src/lib.rs` (design S), `gpu-gate.sh`, `box.sh`, `lease*.sh`, `cards.sh` (3.4); the spin sites if a red points there | L | **Maintainers first**: the fixture recipe published beside the reference sets (answer 3; the sets are public), the contributor gate list (Q4), the fixture path rule (Q2). **Contributor**: the local runner mode (M, answer 10; the maintainers review it), then run, triage, fix Arm-only defects; leaves `gate-ptx-spill` out |
| **P2** a real model runs (the pool invariant: [#16](https://github.com/midagedev/bloomery/pull/16), section 3.6) | Clef-Flash, then Qwen3-30B-A3B and Qwen3.6-35B-A3B, answer from `bloomery-serve` on the Spark with a clean plan and `PoolOver` holding; a placement unit gate holds Σ card + host ≤ pool; the published P2 tape replayed byte for byte once it exists (answer 14) | `crates/placement/src/placement.rs` (`unified: Option<UnifiedPool>`), `workstation.rs` (`unified_of`, `host_of`), `crates/gpu/src/lib.rs` (`census` reads `CU_DEVICE_ATTRIBUTE_INTEGRATED`; free bytes from `MemAvailable` on an integrated device) | M | **Contributor** (`bloomery-placement` is a pure crate: the invariant is testable off the box); the shape is answer 11. **Maintainers first** for the tape clause: the P2 tape and its reference rows (answer 14) |
| **P3** Spark-specific speed | the section 2 term measured; a Spark table with its own ruler; `BLOOMERY_ARCH=sm_121` rows | `ptx-shapes.tsv` per-arch rows (the arch owner, S3, is in: `justfile` `arch`, `tools/arch.sh`), the fork's per-triple key (S2), `crates/threads` P/E split, `timing-card.sh` single-card mode | L | **Maintainers**: S2 (it touches the fork pin). **Contributor**: the measurement, the table |
| **P4** above the pool | a model larger than the pool (V4.1, GLM-5.3-Flash `UD-Q4_K_XL`) answers on one Spark with its overflow on the NVMe tier, the card computing the tier's experts from their slots; the section 7 term measured | the unified tier's slot seam (section 7), `crates/gpu/src/host/nvtier.rs`, the card leg | L | **Maintainers first**: the unified tier and its integrated-device seam. **Contributor**: the read-rate measurement, then runs and extends the seam on the Spark |
| **P5** several Sparks (×2, ×4) | a model plans and runs across N linked Sparks under the section 8 criteria, N = 2 first: split (a), the star, with raw verbs and a pump thread a node (8.5, 8.6); the placement unit gate `nodes_scale` holds the criteria at N = 1, 2, 3 and 4 (8.8) | `crates/placement` (`Machine { cards, tiers, nodes, fabric }`, `nodes_scale`), `crates/gpu/src/host/{mod,tier}.rs` (monotonic words, a `Link` seam), the transport (8.6) | L | **Maintainers first**: single-node P1 (the host-tier protocol on Arm); #16 with its doc line, then `host` and `unified` into `nodes[0]`; the per-node `Machine` and `nodes_scale`; the monotonic-word change and the `Link` seam; the `libibverbs` decision under `just deny` (8.9). **Contributor**: the two measurements of 8.9 (`ib_write_bw` / `ib_write_lat`, then the GPU flag ping-pong probe, which takes N as an argument), then the transport and the N = 2 run |

Order inside P0: gate the import and the 39 functions first (the crate then compiles with every call site still behind
`has_features`), then the predicate split of Q1, then `attn.rs`, then `ops.rs`. Each step can be checked by the cross check
before any Spark run.

## 5. Maintainers' answers (2026-10-10)

1. **`qdot` off x86:** refuse (R) in the P0 pull request, so the crate compiles and whole-on-card models run; scalar
   mirrors (S) in P1, because the fixture tier offloads to the host tier. NEON only when a host-tier gate's wall asks for it.
2. **Fixture identity:** `/models/fixtures` stays the one legal root for now; on another machine, symlink it. A
   root-relative check comes when a contributor needs it, as its own change.
3. **Publishing:** the maintainers publish the reference sets and the fixture regeneration inputs per family on Hugging
   Face, each with its source model's licence file. A family whose source licence does not allow redistribution of its
   `tokenizer.*` keys is not published. The reference sets are up:
   <https://huggingface.co/datasets/midagedev/bloomery-refsets> (public, 17 sets, 6.48 GB, with each base model's
   licence).
4. **The contributor gate list:** the maintainers own it, as one derived command (`local-check.md`): `just affected`
   minus real-file and real-only gates, `gate-ptx-spill` and the timing runners.
5. **Arm evidence:** a contributor's recorded run on the Spark (its own table, with the machine named) is the Arm
   evidence; the maintainers' x86 batch is the gate for main.
6. **P2 golden:** yes. A published Clef-Flash tape, byte for byte, is the P2 criterion once it is published (answer 14);
   until then P2's criterion is "answers, plan clean, `PoolOver` holds".
7. **Toolkit:** CUDA 13.3 for aarch64 (sbsa) is the tested set, not DGX OS's 13.0. `sm_86` PTX through the JIT until P3;
   `sm_121` PTX is a P3 item (`BLOOMERY_ARCH=sm_121`, section 3.2).
8. **64 GB Spark:** best effort, for the models that fit; not a target.
9. **An aarch64 release tarball:** P3 or later.

Added 2026-10-11:

10. **The local runner mode** ([`local-check.md`](local-check.md) open question 3): in `tools/box.sh` as
    `BLOOMERY_BOX_LOCAL=1`, not a separate script. The command string, the guard and the env exports have one owner;
    a second script would copy them. The contributor builds it, as offered on #13; the maintainers review it.
11. **P2's shape:** the cut of M4 (`unified: Option<UnifiedPool>`) and its own `PoolOver`, as #16 builds them. The full
    `MachineSpec` waits for a reader.
12. **Above the pool:** the card computes the NVMe tier's experts from their slots, through the unified tier (section 7).
    The host leg is not the Spark's path for them.
13. **Several Sparks:** one design for N nodes, run at N = 2 first; no code may assume N = 2 (section 8).
14. **The P2 tape, where:** the tape's `Q4_K_M` file and the rows `e2e.py` reads are not public yet
    ([#16](https://github.com/midagedev/bloomery/pull/16)). The maintainers publish the Clef-Flash P2 tape and its
    reference rows in the refsets dataset (answer 3). Until then P2's criterion is "answers, plan clean, `PoolOver`
    holds".

## 6. Not verified

The aarch64 build of P0 and the readings of section 1 are the contributor's records on two DGX Sparks
([#13](https://github.com/midagedev/bloomery/issues/13), [#14](https://github.com/midagedev/bloomery/pull/14)); the
maintainers' checks ran on x86_64. Still open: LLVM 21 and `sm_121` PTX; the host-tier protocol's correctness on Arm
(its spin sites, section 3.1); whether `fixture generate` needs only a source header; fixture fill determinism on
aarch64; the card's read rate over pageable memory (section 7); the NVMe read rate on a Spark; of section 8, the latency of
a direct link, small-message bandwidth in protocol mode, `cuStreamWaitValue32` on a page the NIC wrote, the hit rates, S₁,
BW and the switch hop (8.10). Its link facts (8.1) are sourced.

## 7. Above the pool: the card computes the NVMe tier

**Today, on a machine with PCIe.** The NVMe tier's RAM arena is one anonymous mapping carved into slots, filled by
`O_DIRECT` reads and "lent to the host leg through the model side's `TierSlots` seam" (`crates/gpu/src/host/nvtier.rs:1-9`):
the CPU computes those experts with `qdot`. On a PCIe machine that is the right flow, because the CPU reads host DRAM in
place and a copy to the card would cross the slower bus.

**On one pool.** The card can address pageable memory, an anonymous mapping included, through the host's page tables
(section 1: `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` 1, "coherently accessing pageable memory without
`cudaHostRegister`", and `..._USES_HOST_PAGE_TABLES` 1 `[#13]`; the attributes' text is the CUDA driver API reference's).
So the card can compute the experts in the arena's slots where they lie, with no copy. The attributes promise no rate
and no residency: a slot page the kernel swapped out (16 GiB swap, section 1), compacted or migrated costs the card a
translation fault that the kernel serves. The arena must hold its pages (`mlock` or an equal guard), and the rule below
measures the rate. The CPU computes nothing on the decode path, which matches section 2's timeline: 20 Arm cores beside
48 SMs on the same 273 GB/s add only contention.

**The owner.** The maintainers' unified tier: one residency across the card, RAM and NVMe, whose rounds are landing now.
On an integrated device its card and RAM levels are one pool, so the residency has two levels, the pool and the NVMe
drive. Its slot seam hands a filled slot's address to the card leg instead of the host leg; the maintainers build the
seam and prove on x86 what they can, then the contributor runs and extends it on the Spark (P4).

**The term to measure first, with its decision rule.** One card gemv kernel reading the same expert bytes from (i)
`cuMemAlloc` memory, (ii) `cuMemHostAlloc` memory (the reading asked for on #13) and (iii) an anonymous mapping as the
arena allocates it, with the CPU idle. The suspect term is translation reach: the arena's pages are 4 KiB today, since
THP is in `madvise` mode `[#13]` and `AnonMap` is memmap2's `map_anon` with no `MADV_HUGEPAGE`
(`crates/engram/src/direct.rs:145-150`). Expected: (iii) under (i) by the 4 KiB translations, (ii) near (i)
`[predicted, no Spark number exists]`. The rule follows the copy's cost: a copy into card memory moves 2 × the slot's
bytes through the same pool once per fill (section 3.3's table), and the slower read is paid on every use, so with h
uses per fill the card reads in place while 1/r(iii) − 1/r(i) < 2 / (h · r_pool) `[derived]` (h = 1: down to a third
of (i); h = 10: at 0.83 of (i) or above):
- (iii) inside that band: the card reads the arena in place;
- (iii) outside it: `MADV_HUGEPAGE` on the arena and measure again;
- still outside, and (ii) inside: allocate the arena with `cuMemHostAlloc` (`O_DIRECT` reads into those pages
  `[unverified]`);
- otherwise: copy each filled slot into card memory before its use, on the copy engines.

The NVMe read rate on a Spark is `[unverified]`. Our box's drive reads 6.77 GB/s at queue depth 16
(`docs/cards/qdsweep.card`); that number belongs to that drive and does not carry over. Above the pool the decode is
bound by the tier's misses, whose flow model is the unified tier's.

## 8. Several Sparks (×2, ×4)

This section is the result of a read-only design round (2026-10-11): no code, no run. The facts in 8.1 are NVIDIA's
pages (two forum rows excepted, marked "user"), each with its URL and quoted line. Every figure in 8.4 and 8.7 is
`[derived]` from section 2's cost model, the repo's byte counts and the assumptions stated once in 8.4. Nothing was
measured on two Sparks. The contributor has two Sparks `[#13]`.

### 8.1 Link facts

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
- One decode message (at most 131 KB, 8.4) rides one rail. A prompt batch can stripe over both rails.
- A GPU cannot post to the NIC: without GPUDirect, the paths where a kernel rings the doorbell (IBGDA, NVSHMEM) rest on
  the peer mapping that cuda.html says does not work (that no such path works on GB10 is `[unverified]`). A host thread
  posts every send.

### 8.2 The shape

A node is one unified machine (section 3.6). A remote GB10 does what a tier card does today: it computes routed experts
in place from a handoff and returns rows. `Device::Card(i)` over `all_cards()` already names such a card, so the second
node is a node, not a new device kind. It cannot be a tier with a link cost on today's `Machine`: #16's `PoolOver` sums
every card of `all_cards()` into one pool, so a remote tier would be charged to node 0's pool, and node 1's host-side
rows and arena to node 0's host.

The owner shape. None of it is in the tree (`Machine` is `{ cards, tiers, host }` at main,
`crates/placement/src/placement.rs:223-229`; #16 adds `unified`):

```rust
pub struct Card { /* … */ pub node: usize }       // an index into Machine::nodes; 0 is this process's node
pub struct Machine { pub cards: Vec<Card>, pub tiers: Vec<Card>,
                     pub nodes: Vec<Node>,        // nodes[0]: #16's `host` and `unified` move here
                     pub fabric: Fabric }
pub struct Node { pub host: Host, pub unified: Option<UnifiedPool>,
                  pub nic: u64 }                  // bytes/s a direction, the SoC side (8.1: about 23.7 GB/s)
pub enum Fabric { Alone,                          // N = 1: every machine today
                  Pair { link: Link },            // N = 2, one cable
                  Ring { link: Link },            // N = 3, both ports
                  Switch { link: Link, hop_ns: u64 } } // N = 4: NVIDIA's ×4 is switched
pub struct Link { pub rail: u64 /* bytes/s, one message */, pub round_trip_ns: u64 }
```

- **#16 lands as it is**, with one doc line on `unified` and `host`: "this process's node", so nothing reads them as
  machine-wide. The maintainers then move both into `nodes[0]`: a move-class change in a pure crate, every caller
  through one accessor, before any remote node exists. With one node in the list the plan is exactly section 3.6's.
- `CardOver` stays per card. `HostOver` and `PoolOver` run once per node, over the cards whose `node` matches.
  `HostNeed` gains no field and is evaluated per node.
- **Node 1's figures come from a remote census.** The census reads only this process's CUDA devices. Node 1's `CardSpec`
  and its `MemAvailable` reading must reach node 0 over the control channel before `plan_tiers` runs.
- **One rule term, common to every family.** A tier takes "each layer's next ids after the stage's prefix"
  (`crates/placement/src/placement/workstation.rs:352-362`). Over a link the rule spreads the hot ids to reach the read
  share s that balances `Sh + s·R = λ·BW + (1 − s)·R` per layer (terms in 8.4). A, Sh and R come from the plan's own
  tensor bytes and roles, so no model supplies them. On the box's PCIe the same formula holds with the lane's rate, so
  it is one owner for both machines, not a Spark branch.
- **With the unified tier.** The second node is a compute site with its own residency, not one more tier in the read
  path: reading a remote V4.1 expert costs 16.8 MB / 11.6 GB/s = 1.4 ms, against 16.8 MB / 219 GB/s = 77 µs to compute
  it in place `[derived]`. The tier's ledger gains a node column, (node, id) to state. Node 1 runs its own books over
  its own pool, arena and file copy. On a unified node a card slot and a host segment collapse (section 7), so a node's
  states are pool, arena and file. One new move is pool of node 0 to pool of node 1: a link copy of an expert (0.7–1.4
  ms) on a background lane, at a step boundary, rare. The router's counts go to node 1 once a window for its arena
  eviction and drive the cross-node balance. The unified tier's per-node byte ledger, plan-time rule and prefetch rounds
  are the ones this touches. None needs to change now.

### 8.3 Three ways to split a model

L is the layer count, `t_hop` one message's one-way latency on the link. Per decode token a message carries a hidden
vector or a few rows of them, tens of KB, so the link's bytes are small and its latency is the term `[derived]`. A
prompt is different: its bytes over the link grow with P, which is why today's prompt path moves the experts to the
activations (`crates/gpu/src/host/xstream.rs:5-19`) and no expert ever crosses the link in (a).

| Split | Attention and KV | Per decode token | Per prompt batch of P = 4096 tokens `[derived]` | What scales with N |
|---|---|---|---|---|
| (a) remote expert tiers: plan (b′) (`crates/placement/src/placement/workstation.rs:352-362`): one stage node runs every layer and the head, each other node holds and computes a share of the experts, the link in place of PCIe | on the stage node | about 2·L messages to each remote node, all through the stage node's NIC (today's tier protocol per tier layer: the handoff image out and the rows back, with a go and a done word, `crates/gpu/src/host/tier.rs:6-11`) | per layer the image out (P × n_embd values: q8_1 for V4.1, f32 for GLM and Qwen3.8) and the rows back (one f32 row of n_embd per remote slot, about k/2 a token); out + back a batch: V4.1 11.0 GB (back 10.1), GLM 14.1 GB (back 11.3), Qwen3.8 12.1 GB (back 10.1) | capacity, and the expert bytes each node reads; the stage node's NIC carries every message, so it can become the max resource as N grows |
| (b) pipeline by layers | each node holds its layers' KV | N − 1 messages | P × 16 × n_embd B (the 4 hc streams in f32) at each of the N − 1 node boundaries: V4.1 336 MB, GLM 268 MB, Qwen3.8 168 MB | capacity only: one request still walks the nodes in turn; several requests fill the pipeline |
| (c) expert-parallel: every node computes its share of each layer's experts in parallel | every node runs attention on the same hidden state, so the KV cache is held N times; without that, (c) is (a) | about 2·L messages per node pair, spread over the NICs | per layer every node's partial sums, P × n_embd × 4 B, to each other node | capacity and the per-node byte term, if the expert bytes are balanced |

Decode on one Spark is bound by the pool's bytes (section 2). (a) and (c) split those bytes over the nodes and add the
latency term; (b) adds latency and splits no bytes for one request. (d), tensor parallel, is what NVIDIA documents: 8.5.

### 8.4 Bands `[derived]`

**Assumptions, stated once.** Every figure below holds under them; none was measured on a Spark.
- **BW**, the pool read of our kernels: 219 GB/s, the 80 % of 273 GB/s that section 2 assumes (band 200–235). A plain
  read kernel measured 233 GB/s on `cuMemAlloc` memory (#13); our kernels' own rate, section 2's one term, is unread.
- **λ**, the exposed round trip a layer, link and protocol: 25–70 µs. Terms: two one-way RDMA trips of 3–8 µs `[assumed:
  no direct-link source]`; send serialization 0.5–1.4 µs and rows serialization 4.4–11.3 µs at 11.6 GB/s (51–131 KB);
  two pump wake-ups of 1–3 µs `[assumed]`; two stream-wait passes of 2–5 µs `[assumed]`; the tier graph's copy nodes
  (`c_node` 0.85 µs, [rig-log plan.md:71](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md#L71)). The one measured protocol number in the repo is the host-tier bridge floor,
  21–36 µs a layer on PCIe ([rig-log plan.md:78](https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md#L78)).
- **U**, the usable pool a node for model, KV and scratch: about 115 GB `[assumed]`. Section 1 reads 116.6–116.9 GiB
  (125 GB) of idle `MemAvailable` on a Spark `[#13]` and #16 reads it at plan time, so U is a cautious planning figure.
- **NVMe**: 6.77 GB/s, our box's drive at queue depth 16 (`docs/cards/qdsweep.card`). A Spark's drive rate is
  `[unverified]`.
- N = 4 adds a 1.5 µs switch hop and 1 µs for a further post `[assumed]`.
- The non-byte "rest" of a step (launch chain and latency, 2.6–4.4 ms a token, section 2) is not in the bands. It lowers
  every absolute figure by about 5–15 % and moves the ratios by under 5 % `[derived at the GLM and Qwen3.8 midpoints]`.
- V4.1 and GLM hit rates on one node (ρ₁ 0.6–0.75 and 0.75–0.9) and on two (ρ₂ 0.93–0.98) are `[assumed]`. The V4.1 and
  GLM bands are only as good as they are.
- The (a) figures come from a Monte Carlo of the per-layer window: each routed slot lands on node j with the ownership
  probability the rule picks, and the window is the max over the nodes of `Sh + slots·e` (node 0) or `λ + hop + posts +
  rows ingress + slots·e` (a remote). Node 0's share is optimized: 0.55–0.72 at N = 2 for Qwen3.8 and 0.07–0.15 at N = 4
  for GLM and V4.1. The script moves into `tools/` beside the gate (8.8).

**Model terms** `[derived]` from each model's tensor table (file sizes: `docs/models.md`):

| Term | V4.1 `Q3_K_M` | GLM-5.3-Flash `UD-Q4_K_XL` | Qwen3.8 `UD-Q4_K_XL` |
|---|---|---|---|
| MoE layers L | 40 | 42 | 48 |
| `n_embd` | 5120 | 4096 | 2560 |
| experts / top-k k / expert ff | 384 / 6 / 2304 | 288 / 8 / 2048 | 512 / 10 / 640 |
| bytes an expert | 16.77 MB | 15.20 MB | 3.17 MB |
| R, routed bytes read a token (k × e × L) | 4.03 GB | 5.11 GB | 1.52 GB |
| Sh, shared and dense FFN a token | 0.8–1.5 GB | 1.4 GB | 0.14 GB |
| A, attention, norms, router and head a token | 3.2–5.8 GB | 4.7–8.2 GB | 3.4–4.4 GB |
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
  V4.1 410–490 µs**.
- (b) with one request in flight: the wall is the sum of the nodes plus the hops. With two in flight, the max.
- (c) with one request equals (a) at a fixed s = 0.5, and both nodes carry the dense weights. With two requests,
  attention is data-parallel (one request a node) and the all-to-all runs both ways: `t = (A + Sh + R(2)/2)/BW + L·λ`,
  with R(2) the routed bytes of the two tokens' union.

**Capacity** (pools N × about 115 GB):

| N | V4.1 (needs about 268 GB resident) | GLM (about 203 GB) | Qwen3.8 (about 114 GB) |
|---|---|---|---|
| 1 | no: about 154 GB of routed on NVMe | no: about 88 GB on NVMe | yes, marginal |
| 2 | no: about 43 GB on NVMe ((b): about 19 GB a node) | yes, about 25 GB spare | yes |
| 4 | yes, about 190 GB spare ((b) about 67 GB a stage; (c) adds 6 GB of dense a node) | yes ((c) adds 11.4 GB of dense a node) | yes |

For Qwen3.8 at N = 2 under (a), node 0 holds about 75 GB (4.5 non-routed + 28.8 PLE + about 39 routed + KV) against
about 114 GB on one node, and node 1 about 40 GB.

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
**S₁ is unmeasured on a Spark**) **and the absolute cap that binds:**

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
the serial A is tensor parallel (8.5).

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

### 8.5 Recommendation, and tensor parallel

1. **(a), the star, for one request: plan (b′) with the second Spark as the expert tier and the link in place of PCIe.**
   - It keeps the bits. The join already reads one f32 row per routed slot from the tier
     (`crates/gpu/src/host/tier.rs:19-23`), so moving those bytes over RDMA moves no float.
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

**Tensor parallel, TP = 2, is what NVIDIA documents** (8.1: vLLM `--tensor-parallel-size 2`, TRT-LLM `--tp_size 2`).
`[derived]` `t = (A + Sh + R)/(2·BW) + L·m·λ` with m sync points a layer. At m = 2 on two nodes it reads GLM 25–36 tok/s
and Qwen3.8 49–71 tok/s at one request, which beats (a) where A dominates. It shards every family's attention and dense
kernels (an MLA latent cache replicated), changes the float sum order (a row-parallel all-reduce; column-only splits
avoid that but take 2–4 gathers a layer), and needs an NCCL-class all-reduce every layer. It is the ceiling to remember,
not the first step: a per-family rewrite against "Common code first". Whether to take it instead is the maintainers'
decision.

### 8.6 Transport

**Raw verbs (rdma-core) under our own word protocol, with one host pump thread per node.** NCCL is the fabric check
(`all_gather_perf` per NVIDIA's NCCL playbook), and the fallback only if the probe of 8.9 shows our pump path slower
than NCCL's point to point. Without GPUDirect NCCL stages through host bounce buffers with its proxy thread `[derived
from the cuda.html line; NCCL internals unverified]`, and a send or receive kernel per message would occupy SMs. The
handoff image already lives in `cuMemHostAlloc` pages (`crates/gpu/src/host/tier.rs:561`), which is what 8.1's "what to
use instead" asks for, so the NIC reads the same LPDDR with no copy. The protocol stays `host/mod.rs`'s alone
(`crates/gpu/src/host/mod.rs:56-58`), behind a `Link` seam with two impls: the shared page (PCIe, identity, a move-class
change) and mirrored pages (RDMA). A raw-verbs build adds `libibverbs`, whose licence under `just deny` is
`[unverified]`.

What `crates/gpu/src/host/mod.rs` and `tier.rs` need to cross a node:
1. **Two pages, one a node.** Every word keeps one writer node. The link mirrors a region one way, one RDMA write a
   message.
2. **Monotonic single-writer words.** Today the go counter is added to by the stage and taken back by the tier, and the
   done counter the other way (`tier.rs:14-18`): two writers, at different times. A mirrored absolute write would erase
   the remote decrement. Each becomes a generation the writer stores and the reader waits on with greater-or-equal. This
   is box-provable first on PCIe: the bits are identical and the wait condition changes.
3. **A pump thread a node.** The card cannot post to the NIC. The pump load-acquires the go generation (Arm needs an
   explicit acquire: the memorder row) before posting. The card's system-scope barrier before go
   (`crates/gpu/src/host/mod.rs:20-24`) orders the image before the word.
4. **Data before flag across the NIC.** Register the pages without `IBV_ACCESS_RELAXED_ORDERING`, so write-after-write
   order holds (the `ibv_reg_mr` row), with the flag last on the same QP. Or use write-with-immediate, where the remote
   pump writes the flag after the completion: one more hop, ordered by the completion rule. NVIDIA's perftest ran with
   relaxed ordering on, so its bandwidth is not protocol-mode bandwidth.
5. **The remote is another process.** Node 1 runs a tier server: it loads its experts from its own copy of the file,
   captures its graphs from the go order it is sent at load, and launches a replay on a "replay r of chain c" message,
   which the first image can carry (today the stage process launches the tier's graph, `tier.rs:36-41`). Before the
   plan, the same channel carries node 1's census to node 0.
6. **Loss.** The go deadline already names a lost tier (`tier.rs:48-53`, `CardLost`). A down link also means the stage
   cannot write `RELEASE` to node 1's words, so node 1 needs its own deadline that drains its streams.
7. **Faults and sequence.** The tier's fault-word copy already rides before its signal (`tier.rs:41-43`). The handoff's
   sequence word detects a stale or doubled image across the mirror.

### 8.7 Scaling criteria

One numbered list. Criteria 1–8 are numeric conditions that the unit gate of 8.8 holds on synthetic machines at N = 1,
2, 3 and 4; criteria 9–11 are held by runs and review. The planner gains a predictor, `Plan::predict(&Workload) ->
Prediction { step, per_node: { pool, nic, msgs_per_layer } }`. It reads the plan's own tensor bytes and roles, the card
spec's BW, the nodes' `nic` and the fabric's `Link`, and needs no model file and no per-model code. The workload carries
decode at n in {1, N} and the prompt at P = 4096 with its S₁ (a rate of a card and model pair, so on the workload, not
the card spec). The thresholds come from a machine parameter, `ruler`: the useful-gain line is 2 × the machine's 95 %
interval. The box's default is 2 % (A6000, SD 0.6 %); a Spark's own A/A sets its value (section 3.5). Our ruler is never
baked in.

**Only criterion 1 refuses a plan**, as `CardOver` and `HostOver` do. Criteria 2–7 are **objectives** (6 is a structural
assertion): the planner chooses which nodes it uses, the ownership and later the prompt mode to meet them, and where it
cannot it names why (`NodeIdle { reason }`, a prompt-mode note). An unset `--place` on four Sparks still plans. Criteria
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
   otherwise `NodeIdle { reason }`. A smaller gain cannot be confirmed by an A/B.
8. **N = 1 plans exactly as a single Spark** (gate clause 1).
9. **Prefill.** `pp tok/s @ P = 512 and 4096` at each N is measured and judged like decode, in the machine's own table;
   the prompt predictor of criterion 5 is checked against it. It is a run criterion, not a unit-gate clause.
10. **Host threads.** One pump thread a node posts every send, because no GPU path to the NIC works on GB10 (8.1, 8.6).
    Its cost, two wake-ups of 1–3 µs a layer, is inside λ, so criteria 2 and 6 hold it. No other host thread stands on
    the per-layer path. This replaces the earlier "no host thread" wording.
11. **A lost node ends the request by name**, as a lost tier card does today (`tier.rs:48-53`): the go deadline names
    it, and node 1's own deadline drains its streams (8.6, item 6). No failover.

**Where each fires** `[derived]`, one row per model, N in {1, 2, 3, 4}, λ in {25, 70} µs, A low and high, and the prompt
at S₁ in {1,000, 3,000} tok/s (the pins of the gate in 8.8):

| Model | Decode step (ms), N = 1 → 2 → 3 → 4, λ 25 / 70, A low | Criterion 2 fires | Criterion 7 fires | Criterion 5 fires (prompt) |
|---|---|---|---|---|
| Qwen3.8 | 23.1 → 20.9 / 21.9 → 20.1 / 21.4 → 19.7 / 21.2 | N = 4, λ 70, A low (2.34 ms exposed = 11 %) | N = 3 at λ 70 with A high; N = 4 in every cell (node 3 gains under 2 %) | N ≥ 3 at S₁ 3,000 (busy 51–64 %) |
| GLM | 51.2 → 40.3 / 41.3 → 36.3 / 37.6 → 34.3 / 35.9 | never | never | N ≥ 3 at S₁ 3,000 (77–95 %) |
| V4.1 (pool only) | 38.2 → 30.1 / 31.1 → 27.0 / 28.3 → 25.5 / 26.9 | never | never | N ≥ 3 at S₁ 3,000 (72–90 %) |

Nothing fires at S₁ = 1,000. At N = 2 nothing fires except GLM's prompt, which sits on the 50 % line (4,286 against
8,610): the gate's boundary case, at ≤ and not <.

### 8.8 The gate: `nodes_scale`

None of this exists yet. It is what the maintainers build with the shape of 8.2. `nodes_scale` is a set of unit tests in
`crates/placement`, a pure crate, so `just mac-test` and the box's lib gate both run it.
- **Machines.** `workstation::spark_nodes(n, fabric)` builds synthetic GB10 nodes (`integrated: true`, a 115 GB pool,
  `nic` 23.7 GB/s): N = 1 `Alone`, 2 `Pair`, 3 `Ring`, 4 `Switch`. `Link` is set at λ = 25 and 70 µs and `ruler` at the
  box default.
- **Models.** Three synthetic shapes carry 8.4's terms for V4.1, GLM and Qwen3.8 (L, `n_embd`, k, e, A low and high,
  Sh). No file is read.
- **Expected sets.** The table of 8.7, pinned from a derivation script that moves into `tools/` beside the gate with its
  own self-test, so the pins have one derivation.
- **Clauses:**
  1. `n1_is_today`: every N = 1 plan equals today's plan byte for byte.
  2. `pool_per_node`: criterion 1 refuses V4.1 whole at N = 2 and places it whole at N = 4. No other machine is refused.
  3. `read_balance`: criterion 4 holds on every plan.
  4. `link_share`: criterion 2 holds on the chosen plan. Qwen3.8 at N = 4, λ 70, A low uses at most 3 nodes.
  5. `nic_not_max`: criterion 5 either holds or carries the named prompt-mode note, exactly in the cells of the "fires"
     column.
  6. `messages_o_n`: criterion 6.
  7. `never_slower`: `NodeIdle` names exactly the cells of criterion 7's column, and no chosen plan is slower than its
     N − 1 plan.
- **FAIL-first, one mutant a clause:**
  - M1: `PoolOver` over `all_cards()` into node 0's pool → clause 2 red at N ≥ 2.
  - M2: plan only `nodes[..2]` (a hidden 2) → clause 2 red, because V4.1 is not whole at N = 4.
  - M3: the tier rule by prefix (today's) → clause 3 red at N ≥ 2.
  - M4: λ = 0 in the predictor → clauses 4 and 7 red, because node 3 is used for Qwen3.8.
  - M5: drop the NIC term → clause 5 red, because no note appears in its cells.
  - M6: charge a node term at N = 1 → clause 1 red.
  - M7: `ruler` fixed at 2 % instead of read from the machine → clause 7 red on a synthetic machine whose ruler is 4 %
    (Qwen3.8 at N = 2 then gains under 8 % and must name node 1 idle).

### 8.9 First step

**Maintainers first.**
1. Single-node P1 is the precondition: the host-tier protocol's correctness on Arm is still open (section 6). Two nodes
   add a protocol layer on top of it.
2. Agree the per-node shape of 8.2 and land #16 with its doc line. Then move `host` and `unified` into `nodes[0]`, a
   move-class change, before any remote node.
3. The monotonic-word change in `host/mod.rs` and `tier.rs` (8.6, item 2): a protocol change, the module owner's. It is
   bit-identical on PCIe: one owning-gate run and a same-lease A/B on the box.
4. A `Link` seam where the tier page is written and read: a move-class change, `ptx-scan` identical.
5. Decide the `libibverbs` dependency under `just deny`.

**The contributor's first step needs no engine code.** On two Sparks, one cable, one rail, under a prediction card
(`tools/ref/card.py` format; their own table, the machine named). The probe takes N from the command line from the
start, so nothing in it assumes a pair.
1. **Fabric.** `ib_write_bw -s 65536` and `ib_write_lat -s 16384` and `-s 65536` on one RoCE device, both directions.
   Then `all_gather_perf` per NVIDIA's NCCL playbook.
   - Expected bandwidth 92–109 Gb/s a rail (11.6–13.7 GB/s): NVIDIA's 92.57 and 97.28 and the user's 109.28 (8.1). Below
     90 Gb/s a rail `[a decision line, not a source: the 90 Gb/s line of the N = 4 switch check below, applied to the
     direct link]` the cable, MTU or ordering setup is wrong: fix it before the probe.
   - Expected one-way latency as perftest prints it: 4–10 µs at 16 KB and 8–14 µs at 64 KB `[derived from an assumed 3–8
     µs base]`. Above 10 µs at 16 KB, λ's band of 25–70 µs is low by twice the excess `[derived]`: redo 8.4 before
     trusting criterion 2.
2. **The protocol probe** (a small bin, contributor-sized).
   - Node 0's kernel writes a 16 KB image and a go generation into a `cudaHostAlloc` page registered with `ibv_reg_mr`
     (strict ordering). A pump posts one RDMA write. Node 1's stream waits on its page (greater-or-equal), runs a null
     kernel and writes 64 KB of rows and a generation. Its pump posts back, and node 0's stream wait passes.
   - 10⁵ iterations, p50 and p99. Every 64-byte line of the image carries the sequence, and the probe counts the stale
     lines seen after the flag.
   - **Expected λ_probe p50 = 18–40 µs `[derived]`.** The engine's λ is λ_probe plus 7–30 µs: the tier graph's copy-in
     and copy-out nodes, the stage's handoff, go and wait memops and the join. That gives the 25–70 µs of 8.4. Expected
     stale lines: 0.
   - Decision rule on λ_probe, against the break-even λ* per model (8.4: Qwen3.8 122 µs, GLM 556 µs, V4.1 410–490 µs):
     - **at most 40 µs** (engine at most 70 µs): (a) as predicted; Qwen3.8 1.05–1.10× at one request; GLM and V4.1 are
       decided by fit.
     - **40–120 µs**: (a) only for the models that do not fit one Spark (GLM, V4.1). Qwen3.8 sits near its λ* of 122 µs,
       so it stays on one node.
     - **over 120 µs**: the pump or wake-up path is wrong; fix it before any engine work.
     - **stale lines over 0**: switch to write-with-immediate and measure again.
3. Then the single-node term section 2 names (sustained BW), which sets every band above.

**What carries from N = 2 to N = 4 unchanged:** the pump threads, the strict-order registration, the generation waits
and the stale-line check, the pair λ, each node's NIC cap (the same SoC links) and the per-node census over the control
channel.

**What N = 4 adds:**
- **A switch health check first.** `ib_write_bw` through the switch at 90 Gb/s a rail or more, before anything else. The
  switch's own settings can hold a link far under line rate, and no source here measures a switched Spark
  `[unverified]`.
- **The switch hop**, measured as λ_switch − λ_pair. Expected +0.5–2 µs `[assumed]`.
- **A fan-out and fan-in mode in the probe.** Node 0 posts to 3 peers and takes 3 rows messages. Expected λ_fanout =
  λ_pair + 2 µs of posts + 3 × 61 KB / 23.7 GB/s ingress (7.7 µs) ≈ λ_pair + 10 µs `[derived]`.

### 8.10 Still open

- GLM's A band (4.7–8.2 GB a token) is wide: the file minus routed leaves about 3.5 GB that the tensor shapes do not
  explain. A header read of the `UD-Q4_K_XL` file's non-routed tensors closes it on a Mac.
- V4.1's hit rates ρ₁ and ρ₂ are assumptions. A Mac-only replay of the route trace at the two room sizes could replace
  them before any Spark run.
- Whether `cuStreamWaitValue32` on a `cudaHostAlloc` page written by the NIC behaves on GB10 is `[unverified]`. The
  probe answers it.
- Unverified besides: a latency for a direct link, small-message (16–64 KB) bandwidth in protocol mode (strict ordering,
  one message in flight; at least 4.8–5.6 µs of serialization on one rail plus the one-way latency `[derived]`), the
  switch hop, BW, S₁, and the Spark drive's NVMe rate.
- Whether to take tensor parallel first instead of (a) is the maintainers' decision (8.5).
