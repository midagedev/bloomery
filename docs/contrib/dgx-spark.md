# Porting bloomery to the NVIDIA DGX Spark

A map of what stands between this tree and "builds and passes the fixture-tier gates on a DGX Spark", written for a
person who owns one. Every claim about the code cites `path:line` as of main `6b630e5e`; a thing that was not seen is
marked `[unverified]`, a number that is computed here is marked `[derived]`. Nothing in this file was run on a Spark.

Two earlier documents already sized this work: [`docs/research/machine-axes.md`](../research/machine-axes.md) (the
`machineaxes` memo, 2026-09-27; its round names `sparkprep` S1–S4, `archkey`, `isarefuse` and "M1–M3" are used below) and
[`docs/rebuild.md`](../rebuild.md) (the "machineaxes" result bullets, lines 197–200). Their line numbers are stale; the
citations here are re-read. Related: [`new-model.md`](new-model.md) (the other contribution track),
[`local-check.md`](local-check.md) (the off-box check design; this file keeps only what is Spark-specific).

**Where it stands.** None of the Spark-prep rounds has landed in this tree:

| Round (memo name) | What | In the tree today |
|---|---|---|
| `sparkprep` S1: cfg guards on the x86 bodies | `qdot`, `deepseek2/attn`, one bin | done in [#14](https://github.com/midagedev/bloomery/pull/14) (P0): every AVX2 item under `cfg(target_arch = "x86_64")`, design R off it |
| `sparkprep` S2: per-triple `extra-rustflags` | fork patch | not done: `.cargo/cuda-oxide.toml:5` is a flat list; ledger row 32 (`docs/upstream/nvlabs-ledger.md:42`) is open and the fork still reads a flat `Vec` (`cargo-oxide/src/commands/context.rs:19,311` at fork rev `4da6c13`) |
| `sparkprep` S3 / `archkey`: one owner of `--arch` | 136 recipe lines | not done: `grep -c -- '--arch sm_86' justfile` is 136, plus `tools/gate.sh:35` |
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
`llc` are available is `[unverified]`.

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
longer than on the box. The `machineaxes` round reached 2.47× for the A6000 (`docs/rebuild.md:200`).

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
  with 16.7 GB left for OS, KV and runtime `[derived; GB against GB, DGX OS's own use unknown]`.

**The one term a run must settle, with its expected value:** the sustained read rate of our decode gemv kernels on the
LPDDR5x with the CPU idle. Expected 200–235 GB/s (73–86 % of 273) `[assumed, no Spark number exists]`. Decision rule: at or
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
| `.cargo/config.toml:7` | `-C target-cpu=znver3` under `[target.x86_64-unknown-linux-gnu]` only | none: plain `cargo` on aarch64 builds for the generic target; add `target-cpu=native` (or `gb10` with LLVM 21) locally for speed | S |
| `.cargo/cuda-oxide.toml:5` | the same flag, flat, for every `cargo oxide` build on any host | an unknown CPU name is a warning in LLVM `[unverified]`; until the fork takes a per-triple key (S2), a Spark owner edits the local copy to `target-cpu=native` (`docs/BUILD.md`, "The CPU flag"); `tools/check-rustflags.sh:24-28` then fails by design, so skip it on aarch64 | S |
| `crates/threads/src/lib.rs:786-850` | pool topology from sysfs (`thread_siblings_list`, `cache/index3/*`); with no L3 file it falls back to one group (`:798`, `:806-819`) | none to compile. Default thread count is the primary-core count (`:422`: 20 on GB10) with equal chunks (`:711`), so the 10 E-cores gate every barrier; NVIDIA's "two clusters" layout is not modeled | none for P0–P2; M in P3 |
| `crates/placement/src/placement/workstation.rs:132`, `paged_drop.rs:181` | `ROW_PAGE` and `PAGE` are 4096, commented "on x86_64" | the DGX OS base page size is `[unverified]`; read `sysconf` or refuse by name on another size | S |
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

- **Targets today.** Recipes pass `--arch sm_86` (136 lines, `tools/gate.sh:35`); `tools/ptx-scan.sh:141`,
  `tools/lds-scan.sh:147`, `tools/sass-scan.sh:59` default to it. The kernels are SIMT code with no generation-specific
  instruction (the three `ptx_asm!` sites are `cvta`, `ld.global`, `max.f32`: `crates/gpu/src/linear/delta.rs:121-157`,
  `flash_gqa_prefill.rs:289`; `fault.rs:714` not read).
- **Documented state.** `docs/cuda-oxide.md:5`: "Everything here runs on Ampere (sm_86) only." The toolchain ledger
  (`docs/upstream/nvlabs-ledger.md`) has no `sm_121` or Blackwell row; its only Spark-related entry is row 32 (the flat
  `extra-rustflags`). So no defect of the pinned cuda-oxide on Blackwell is known, because none has been tried.
- **P0–P2 need no new target.** The binary carries `.target sm_86` PTX and the driver JIT-compiles it at load; the release
  states it (`tools/release/README.release.md:15-16`: "the archive carries sm_86 PTX, which the driver compiles for the
  card on first start"), and the loader says it for Blackwell: "A payload built for a standard pre-Blackwell target, such as
  `sm_86`, may instead be converted to PTX and JIT-compiled by the driver on Blackwell"
  (`cuda-host/src/embedded.rs:58-62` at fork rev `4da6c13`, `Cargo.toml:30`). First JIT cost on Grace cores is
  `[unmeasured]` (10–25 s measured on the box, `README.release.md`).
- **`sm_121` for P3.** The fork's target table has `capability: 121` (`cuda-target-spec/src/lib.rs:382,417,447`) and its
  selection tests name `sm_121a`/`sm_121f` (`cuda-oxide-codegen/src/target/tests/selection.rs:162-179`). Whether LLVM 21's
  NVPTX backend accepts `sm_121` is `[unverified]`; so is a `cargo oxide doctor` pass on aarch64. The fork's `tcgen05` path
  is datacenter-Blackwell only (`rustc-codegen-cuda/README.md:107-108`: "need `sm_100a`, which a consumer `sm_120` lacks"),
  and we use no such instruction.
- **Pins that follow the compute capability.** `tools/ref/ptx-shapes.tsv:6` pins spill and `jit_local` from "sm_86, JIT on
  the RTX 3090"; `tools/recipes.py:1960-1965` says the JIT columns follow the capability; `tools/ptx-scan.sh:104` hard-codes
  the sm_86 occupancy column. So `gate-ptx-spill` and every occupancy-derived pin are red or meaningless on cc 12.1 until
  the maintainers add per-arch rows (`archkey`, memo M1). A Spark contributor leaves them out.
- **Bit gates.** The memo concludes the bit gates have no generation dependence by structure (they compare against oracle
  files) and names two residual risks: `ptxas` contracting `.rn`-less mul-then-add differently, and the `ex2.approx`-class
  approximate instructions in about 30 places (`docs/rebuild.md:198`, `docs/research/machine-axes.md:10-16`). The proof is
  one bit-gate pass on the Spark (`gate-gpu-e2e`'s count pin and the card family gates). The repo has only ever run
  cc 8.6 (`tools/recipes.py:1960`: "both cards here are sm_86").
- **Device pick.** The real-tier gate plan opens the card by the names in `CARDS` and refuses any other card by name
  (`crates/gpu-gates/src/bin/shared/gate_card.rs:8-9, 38-47`: "none of this workstation's cards"). The fixture tier plans on
  `--place a`, "the largest visible card as the device reports itself" (`gate_card.rs:4-6`) and is not affected. The real
  tier is the maintainers' tier regardless (section 3.4).

### 3.3 Unified memory

Placement assumes a discrete card, a PCIe host tier and an NVMe tier (`crates/placement/src/placement.rs:223-229`:
`Machine { cards, tiers, host }`; there is no pool field, `grep -i unified crates/placement/src` finds nothing).

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
| H2D promotion over PCIe (`cuMemcpyHtoDAsync`, `crates/gpu/src/host/xstream.rs:1156,1194`; the staging rings `:267,694,1493`) | vanishes as a PCIe cost; a copy from pageable memory into a `cudaMalloc` buffer is still a DRAM-to-DRAM copy at half the pool's rate (reads and writes both cross it: 21.2 GB in ~0.16 s `[derived]`), negligible beside the file read |
| Router D2H handoff and the host service protocol (`host/mod.rs:1-60`) | unused when no layer has host experts; works on `cuMemHostAlloc` pages (`host/tier.rs:561`, `xstream.rs:267`), the allocator NVIDIA recommends; correctness on Spark `[unverified]` |
| CPU expert compute | stays possible but pointless: 20 Arm cores against 48 SMs on the same bytes. It is the path the fixture tier exercises (3.4) |
| The pool's bandwidth | stays, as the single term (section 2) |
| Page cache of the model file plus the device copy | stays and doubles the footprint until dropped; `nvtier.rs:12-30` already drops pages after use (`MADV_DONTNEED`, the `release_union` path) for the NVMe tier |
| NVMe → DRAM stream for models above the pool | stays; its rate on a Spark is `[unverified]` |
| KV cache bytes | stay, and they come out of the same pool |

**Smallest first placement: everything on the card, no host tier, no NVMe tier.** Two ways in, both generic already:
`--place a` is `Pick::Rank(0)`, "the largest visible card" (`crates/placement/src/placement/devices.rs:219-224`), and an
unknown card takes the census path with its own total and free bytes (`devices.rs:98-123`); the Qwen3 and Qwen3.6 seat
opens "CUDA device 0" (`crates/gpu/src/lib.rs:2124`, `docs/BUILD.md:88`). `--place gate` needs a card named 3090
(`devices.rs:221`) and is refused. (`docs/BUILD.md:77` still says `a` needs a card named A6000; that is stale.)

Models by size, in the order to try them (`docs/models.md:12,31-34`, `docs/BUILD.md` table): Clef-Flash 4.8–9.7 GB,
Qwen3-30B-A3B `Q4_K_M` 18.6 GB, Qwen3.6-35B-A3B `Q4_K_M` 21.2 GB; then Qwen3.8 `UD-Q4_K_XL` 111.3 GB and GLM-5.3-Flash
`UD-Q2_K_XL` 101.3 GiB (the size chosen "to fit a DGX Spark", `docs/plan-ledger.md:359`) if the first `MemAvailable` reading
allows. A 64 GB Spark (section 1; announced, not yet shipping) holds only the first three.

### 3.4 Running the gates off our machine

What a contributor's host must not assume, and the smallest change for each (the design of the one off-box command is
`local-check.md`'s):

| Assumption | Where | Smallest change | Size |
|---|---|---|---|
| Work runs over ssh on host `ws`, in `~/repo/<tree>`, after an rsync | `tools/box.sh:42,43,108,225` | a `BLOOMERY_BOX=local` mode: run the recipe's command in the same tree, sourcing `$BLOOMERY_ENV` instead of `~/bloomery-env.sh` | S |
| Locks and holds under `/root` | `/root/bloomery-cpu.lock`, `-<owner>-hold` (`box.sh:10-11`, `tools/ref/lease-probe.sh`), `/root/bloomery-gate*.lock`, `-v41-load.lock`, `-batch.gpuhold` (`tools/gpu-gate.sh:48-50`); 41 `/root/` occurrences in `box.sh` 5, `gpu-gate.sh` 8, `lease-probe.sh` 4, `lease.sh` 5, `gate-batch.sh` 11, `justfile` 7, `ref-paths.sh` 1; in all 30 files under `tools/` name `/root/`, 17 of them under `tools/ref/` (the model profiles `tools/ref/models/*.sh` that `ref-paths.sh` sources, the dump scripts, the C++ harnesses) | one `BLOOMERY_LOCK_DIR` (default `/root`). Precedent: `gpu-gate.sh --test-locks DIR` already moves the three locks and the hold file for its self-test (header, line 17) | M |
| Two named cards | `tools/ref/cards.sh:31-35` matches the exact driver names `NVIDIA GeForce RTX 3090` and `NVIDIA RTX A6000`; `gpu-gate.sh:697-726` picks `3090\|a6000\|any\|both`; a Spark has neither, so both UUIDs stay empty | `BLOOMERY_GATE_CARD=local`: one lock under the lock dir, no name match, same bound and exit codes (`tools/gate.sh` stays the owner of the code). `nvidia-smi` may print `[N/A]` for GB10 memory fields (`[unverified]`; our census uses the driver API, `gpu/src/lib.rs:1789`) | M (`gpu-gate.sh` is 1,021 lines; its self-test needs cases) |
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
(`dump-ref-fixture`, `justfile:609`) from the C++ harness the maintainers build (`just build-ref`). Those sets are what the
maintainers must publish, together with how to regenerate the fixtures. The fixture files are about 49 GB in all, measured on the box (glm5next 8 GB, qwen38 11 GB, v41 16 GB, v41-r8
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
- A witness for this machine: card name and power limit (`timing-card.sh:130`; `power.limit` may be `[N/A]` on GB10,
  `[unverified]`), clock-event counters, the Xid count (`:27`, needs the kernel journal), and, because the pool is shared,
  CPU load, `MemAvailable` and page-cache state at the start and end of every arm (NVIDIA's `drop_caches` line is the
  debugging aid, section 1). `cpu-freq` reads `scaling_cur_freq`; its presence on Arm is `[unverified]`.
- Every number carries `tok/s @ n=N, depth D, card` (`AGENTS.md`) plus the machine line: GB10, memory, DGX OS, driver,
  CUDA toolkit, clocks.
- The ruler is theirs: the 0.6 % same-binary SD is our box's (`AGENTS.md`, "Know the ruler"). The first Spark run is an
  A/A of at least six rounds, and `tools/ref/card.py` then sets the round count for any claim.
- A reference engine arm, interleaved in the same lease (`AGENTS.md`, "the line to beat"), is a build the contributor makes
  for aarch64; no repo recipe does it today.

## 4. Phased plan

Sizes are S ≤ a day, M a few days, L a week or more of work for someone who knows the code. "Maintainers first" is what
must exist before a contributor can start; "contributor" is what the Spark's owner does.

| Phase | Done when | Files | Size | Who |
|---|---|---|---|---|
| **P0** compiles on aarch64 (done, [#14](https://github.com/midagedev/bloomery/pull/14)) | `cargo check` of the host chain passes on aarch64 Linux (on a Spark, or the cross check of 3.1 from any machine), then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe` on the Spark; and on x86 nothing moved: `just ptx-scan` equal and `gate-qdot`, `gate-ops` green | `crates/qdot/src/lib.rs`, `crates/model/src/arch/deepseek2/attn.rs`, `crates/model/src/ops.rs:5529`, `markov-accept.rs`, the two test files; `tools/mac-check.sh` triple; a local `.cargo/cuda-oxide.toml` | M | **Contributor** writes it. **Maintainers first**: decide Q1 (R or S) and answer Q5; **maintainers after**: run the x86 landing batch on the PR (the box is the only x86 judge) |
| **P1** fixture gates pass | the host gates whose inputs are in process or published, and the three fixture families' gates, are green or each red is a named, Arm-only clause; a first bit-gate pass on cc 12.1 | `crates/qdot/src/lib.rs` (design S), `gpu-gate.sh`, `box.sh`, `lease*.sh`, `cards.sh` (3.4); the spin sites if a red points there | L | **Maintainers first**: local runner mode (M), reference sets and fixture recipe published (S–M, box time already exists), the contributor gate list (Q4), the fixture path rule (Q2). **Contributor**: run, triage, fix Arm-only defects; leaves `gate-ptx-spill` out |
| **P2** a real model runs | Clef-Flash, then Qwen3-30B-A3B and Qwen3.6-35B-A3B, answer from `bloomery-serve` on the Spark; a placement unit gate holds Σ card + host ≤ pool | `crates/placement/src/placement.rs` (a pool field), `workstation.rs` (a Spark instance, `ROW_PAGE`), `crates/gpu/src/lib.rs:1974-2000` (free bytes from `MemAvailable` on a unified machine) | M | **Contributor** (`bloomery-placement` is a pure crate: the invariant is testable off the box); **maintainers first**: agree the `MachineSpec { unified }` shape (memo M4) and the golden to compare against (Q6) |
| **P3** Spark-specific speed | the section 2 term measured; a Spark table with its own ruler; `--arch sm_121` rows | `justfile` + `tools/gate.sh` arch owner (S3), `ptx-shapes.tsv` per-arch rows, the fork's per-triple key (S2), `crates/threads` P/E split, NEON `qdot` kernels only for models above the pool, `timing-card.sh` single-card mode | L | **Maintainers**: `archkey` and S2 (they touch every recipe and the fork pin). **Contributor**: the measurement, the table, the NEON work if the NVMe tier matters |

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
   `tokenizer.*` keys is not published.
4. **The contributor gate list:** the maintainers own it, as one derived command (`local-check.md`): `just affected`
   minus real-file and real-only gates, `gate-ptx-spill` and the timing runners.
5. **Arm evidence:** a contributor's recorded run on the Spark (its own table, with the machine named) is the Arm
   evidence; the maintainers' x86 batch is the gate for main.
6. **P2 golden:** yes. A published Clef-Flash tape, byte for byte, is the P2 criterion.
7. **Toolkit:** CUDA 13.3 for aarch64 (sbsa) is the tested set, not DGX OS's 13.0. `sm_86` PTX through the JIT until P3;
   `sm_121` PTX is a P3 item.
8. **64 GB Spark:** best effort, for the models that fit; not a target.
9. **An aarch64 release tarball:** P3 or later.

## 6. Not verified

The aarch64 build of P0 is the contributor's record on a DGX Spark ([#14](https://github.com/midagedev/bloomery/pull/14));
the maintainers' checks of it ran on x86_64. Beyond it: the ISA beyond Armv9.2-A (NEON, SVE2,
dot-product, i8mm); a 13.3 aarch64 toolkit and LLVM 21 `sm_121`; PTX 8.7 JIT on the R580 driver for cc 12.1; DGX OS page
size; `nvidia-smi` fields on GB10; `cuMemGetInfo`/`MemAvailable` readings on a booted Spark; sysfs `cache/index3` on GB10;
host-tier protocol correctness on Arm; whether `fixture generate` needs only a source header; fixture fill determinism on
aarch64; the reference-set size.
