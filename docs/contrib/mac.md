# Porting bloomery to Apple Silicon (Metal)

This is the map and the design for running bloomery on an Apple Silicon Mac (aarch64-apple-darwin, Metal GPU, one
unified memory). It says what builds on a Mac today, which work is left, in what order, and what proves each step. It
changes no code. The target machine is the M5 Ultra, and the goal is the fastest decode and prefill on the M5
generation, so the Metal backend (Part C) is the main part and the CPU path (Part B) is cut to what the Metal work needs.

**How to read the numbers.** A claim about the code cites `path:line` at main `8d74220bb`. A number computed here is
marked `[derived]` with its inputs. A thing that was not seen is marked `[unverified]`. A hardware or toolchain fact
from the web carries its URL and one line copied from that page; a fact read from a local file names the file. Every
URL was opened in this round. "Measured" means a command run in this round on the one Mac that was available, an
Apple M1 Pro with 32 GiB (the machine facts are in section 2); every probe's log is under
`target/mac-portmap/` of the worktree (gitignored). No speed number is measured here: `AGENTS.md` says a timing comes
from a runner in `tools/ref/`, and none runs on a Mac. The only speed numbers are byte-over-bandwidth ceilings marked
`[derived]`.

**What changed since this map was written.** Parts A to F are the map and design of 2026-10-21 (main `8d74220bb`, with
the M2.0 measurements of the same day) and keep that date. The Metal backend has landed since. Section 1 says what runs
on Metal today, at main `d916e20ac`; where a later part says something does not exist yet, section 1 and the status line
at the head of that part win. The user-facing steps (install, build, serve, flags, numbers) are in
[`../mac.md`](../mac.md).

Related documents: [`dgx-spark.md`](dgx-spark.md) (the port map this one is modelled on; its Appendix A is the P0 site
map for aarch64 Linux), [`../research/tierengine.md`](../research/tierengine.md) (item (h) is unified memory on GB10),
`AGENTS.md` ("Derive first", "Never").

## 1. Where the port stands

| Phase (issue #593) | Done when | State (main `d916e20ac`) | Left |
|---|---|---|---|
| **M0** native host build | the host chain builds and tests on aarch64-apple-darwin; every Linux-only site is behind one platform module; on x86 Linux nothing moved (`just ptx-scan` equal) | `crates/os` (`bloomery-os`) holds the Linux-only libc calls of the host chain beside their macOS bodies or a named `OsError::Unsupported`, and `threads`, `gguf`, `engram` and `model` call it (`cd40ceac9`, `000a5d52d`, 2026-10-21); `cd40ceac9`'s message records `tools/mac-check.sh test` at 16 crates, 1577 passed, 0 failed. The census in Part A (14 of 21 crates built; 1477 passed, 0 failed, 210 ignored in 52 test binaries; 45 compile errors at 38 source lines in 10 files) is of the tree before it | a fresh count; this page did not rerun it `[unverified]` |
| **M1** CPU path | NEON for `qdot`; the CPU engine behind `serve::Engine` | 2026-10-21, not rechecked here: `qdot` has a scalar mirror for every kernel, so it builds off x86 once `threads` builds; a NEON form of the two generic kernels was prototyped and is bit-identical to the scalar mirror on 2400 rows (Part B); no CPU engine implements `serve::Engine` | off the critical path to a fast Metal decode (Part E) |
| **M2** Metal backend | one model's decode set runs on Metal and a Mac gate holds it | `bloomery-device` (the device contract) and `bloomery-metal` (its Metal backend) landed on 2026-10-22 (`c16dd696c`). The `qwen35moe` family runs on it, in MSL kernels, with batched prompt passes and, on a GPU with matrix units, the tensor-op GEMMs for the prompt (`aaf14ad24`, `d91e4000e`, 2026-10-25). Files that run and files refused by name: section 1.1. Gates: `just lab-metal` and `just lab-metal-e2e`. For scale, as counted on 2026-10-21: the CUDA side has 376 `#[kernel]` fns and 90 `#[cuda_module]` blocks, and a Qwen3-MoE decode step launches 15 distinct kernels (Part C) | Qwen3.6 `UD-Q4_K_XL` (section 1.1); image input, a second slot and speculative decoding ([`../mac.md`](../mac.md)) |
| **M3** a Mac gate tier | gates run on a Mac; `AGENTS.md` "Never" says so | `tools/mac-gate.sh` (the Metal lock, the bound, the refusals; `ee4ee6cb2`, 2026-10-22) runs the Metal gates through `just lab-metal` and `just lab-metal-e2e`. Both are `lab-` recipes: the box has no Metal device, so no landing batch selects them. The static tier and the pure crates' tests run as before (`just mac-static`, `just mac-test`). No timing runner with a witness exists on a Mac, so a Mac speed number is a development-loop number | a Mac timing runner and its witness (Part D, F6) |

The order that reaches a fast Metal decode soonest is M0, then M2, then M3; M1 moves out of the critical path (Part E
gives the reasons).

### 1.1 What runs on Metal today

At a build after v0.3.9 two architectures run on Metal, `qwen35moe` and `qwen3moe`. The open takes mat-vec weights of type Q4_K, Q6_K or
MLX's affine 4-bit (`MLX_A4`), the embedding as Q4_K or `MLX_A4` and the head as Q6_K or `MLX_A4`
(`crates/metal/src/arch/qwen3moe/arena.rs:333,617,624`). Any other type at those sites is refused by name, with the
tensor's name. The gate verifies the MLX 4-bit path on the 4-layer fixture splits of Qwen3.6-35B-A3B and
Qwen3-30B-A3B: their logits at positions 0 to 3 match mlx-lm's in f32 within 0.01. It opens no full folder. A full
folder opens with the same code path. A GGUF file of these types runs, unverified.

| File | State | Evidence |
|---|---|---|
| Qwen3.6-35B-A3B, an MLX 4-bit folder (`--hf` or `-m`; `prepare` is optional) | verified on the 4-layer fixture split | [`../mac.md`](../mac.md) (numbers, conditions) |
| Qwen3.6-35B-A3B, Q4_K_M GGUF | runs, unverified | [`../mac.md`](../mac.md) |
| MiniCPM-V 4.7 35B-A3B, Q4_K_M | runs, unverified, **text only**: its `mmproj` is not read on Metal. Its vocabulary is 248,144; the open refused it before PR #805 (`58d583fa6`) with `qwen35moe arena: a vocabulary of 248144; the head reads 248320`, and the v0.3.9 tag does not hold that commit | PR #805's comment: `generate_metal` on a real file, rc 1 at the base and rc 0 with the change |
| Ornith-1.5-35B, Q4_K_M | runs, unverified, text only, on v0.3.9's `crates/metal` too. It has Qwen3.6's shapes and a next-token layer beyond the trunk; the open reads the trunk's layers only (`crates/model/src/arch/qwen35moe/spec.rs:9`) | PR #805's comment: rc 0 at the base and with the change |
| Sage-A1.5, Q4_K_M | runs, unverified. Not run on Metal in the sources of this page. Its file has Qwen3.6's shapes, plans the same weight bytes (`--plan` on the CUDA path) and carries a next-token layer too ([`../models.md`](../models.md)), so it is expected to open as Ornith does | none yet |
| Qwen3.6 `UD-Q4_K_XL` | runs, unverified, text only, on a build after 0.3.9: Q8_0 and Q5_K at the mat-vec, embed and head sites, f32 `ssm_alpha`/`ssm_beta`, and a Q8_0 shared expert beside Q4_K/Q5_K routed stacks | #858 and its follow-up: `--plan` from 350 refusals (PR #806) to 70 to 0; on an M1 Pro (32 GB) the real file's prompt pass and stepped run give the same 32 greedy ids, and llama.cpp's CPU logprobs at those 33 steps put the same id first at every step |
| Qwen3-30B-A3B-Instruct-2507, Q4_K_M | runs, unverified, text only (`qwen3moe`: attention layers only, no shared expert). The e2e gate's qwen3moe row opens the MLX split | PR #856: the fixture row; on an M1 Pro (32 GB) the real file's prompt pass and stepped run give the same 32 greedy ids |
| Qwen3-Coder-30B-A3B-Instruct, MLX 4-bit folder (`--hf mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`) | verified, text only, on a build after 0.3.10, on Qwen3-30B-A3B's 4-layer fixture split: it has those shapes (`qwen3moe`), and the gate does not open its folder (the folder names no BOS and adds none) | PR #886: on an M1 Pro (32 GB), `generate_metal --hf` at `6a05f2cf6` opened the folder from the hub cache (all 13 files' hashes checked) and wrote sensible Python; a `bloomery-serve-metal --hf` chat answered with a correct `is_prime`. The tool-call PR: a `get_weather` request returned `tool_calls` with `{"city":"Seoul"}`, and the tool's result came back as the answer |

The refused file is the next to open; the design is [`../research/metal-all-models.md`](../research/metal-all-models.md) (PR #802).

The one rate in PR #805's comment: an Apple M1 Pro with 32 GB, macOS 27.0.1, `generate_metal --prompt "The capital of
France is" -n 32`, greedy, with a download running on the same Mac. MiniCPM-V and Ornith each decoded at 43.4 tok/s.
That is a development-loop number, not a measurement of the engine: the Mac has no timing runner. It also shows that a
32 GB Mac opened and ran two Q4_K_M files of 35B-A3B shape.

### 1.2 Before opening a file: `generate_metal --plan`

`generate_metal --plan [--ctx C]` reads only the file's header and tensor table and loads no weights. It prints every
refusal the open would raise, not only the first, then the bytes the open would allocate, the device's working set and
the host's free memory. It exits 0 when the header's checks pass, 3 when it is refused and 64 when the header cannot be read; exit 0 does
not prove the load fits the device or the host at run time. The data section may be truncated: a 24 MB cut of a Hub file is enough. In PR #806 the open's checks move into one pure
function over the header and tensor table, and the open stops at the first refusal with the text it had before.

PR #806's evidence, on the lead's M1 Pro (32 GB), with `--plan` on 24 MB header cuts of Hub files: the MiniCPM-V and
Ornith Q4_K_M files print `plan: opens`; Qwen3.6 `UD-Q4_K_XL` prints `plan: refused, 350 items`; Qwen3-30B-A3B is
refused at the architecture.

### 1.3 The Metal toolchain, and which kernel path is tested

`crates/metal/build.rs` runs `xcrun -sdk macosx metal --version` (`build.rs:184-191`). If it answers, the build compiles
the kernels into a metallib and the runtime loads that. If it does not, the build embeds the kernels' source and the
runtime compiles it when the device opens (`crates/metal/src/library.rs:199-215`). The tensor-op GEMMs that a GPU with
matrix units takes (Metal 4; the M5 in [`../mac.md`](../mac.md)) are compiled from source at the open on both paths
(`library.rs:207-214`).

One maintainer Mac has the Metal Toolchain; the others have none (`xcrun -sdk macosx metal --version` prints
`error: cannot execute tool 'metal' due to missing Metal Toolchain`). The metallib path builds on the Mac that has it.
Every gate and every number on this page ran the source path. **The metallib path's gates have not run on any of the
maintainers' Macs.** Its `metallib_equals_source` clause compares Q4_K/Q6_K mat-vec cases and an embedding case
between the two libraries, word for word; a build without the toolchain reports it as a skip. Other entries are not
compared.

A tester with the toolchain installed (`xcodebuild -downloadComponent MetalToolchain`) can help: run `just lab-metal`
and report its output. The kernels gate prints a `metal-lib: path=metallib` or `path=source` line; its
`metallib_equals_source` clause gives a verdict only on a build that made a metallib.

## 2. The machines

The measured machine is the only Mac used in this round; the M5 rows are Apple's own statements, with the line copied.
No M5 machine was available in this round, so nothing about the Neural Accelerators is measured here; a contributor's
M5 Ultra numbers are in [`../mac.md`](../mac.md).

| Fact | Value | Source |
|---|---|---|
| Chip, memory, cores | Apple M1 Pro; `hw.memsize` 34359738368 B (32 GiB); 10 cores, 8 performance + 2 efficiency (`hw.perflevel0.logicalcpu` 8, `hw.perflevel1.logicalcpu` 2); 16 GPU cores | measured: `sysctl`, `system_profiler SPDisplaysDataType`; `machine-probe.log` |
| OS, Metal | macOS 27.0 (26A428); Metal 4 supported; GPU family `apple7`, not `apple8`/`apple9` | measured: `sw_vers`, the Swift `supportsFamily` probe (`probes/dev.swift`, `machine-probe.log`) |
| Page size, cache line | 16384 B (`hw.pagesize`), 128 B (`hw.cachelinesize`); L2 12582912 B for the performance cores, 4194304 B for the efficiency cores | measured: `sysctl` |
| CPU ISA | `hw.optional.arm.FEAT_DotProd` 1, `FEAT_I8MM` 0, `FEAT_BF16` 0, `FEAT_SME` 0 | measured: `sysctl` |
| GPU memory limits | `hasUnifiedMemory` true; `recommendedMaxWorkingSetSize` 26800603136 B (0.78 of `hw.memsize` [derived]); `maxBufferLength` 20100448256 B; `maxThreadgroupMemoryLength` 32768 B; `iogpu.wired_limit_mb` 0. The working set is 0.78 × `hw.memsize` on all three Macs measured: two M1 Pro 32 GB (24.96 GiB each) and an M4 Pro 48 GB (37.44 GiB); above 48 GB the fraction is `[unverified]` | measured: Swift `MTLDevice` probe, `sysctl iogpu.wired_limit_mb`; all three Macs: [`../research/metal-offload.md`](../research/metal-offload.md) §1.1 |
| Missing tools | `flock`, `timeout`, `gtimeout`, `setsid` are not on this Mac | measured: `which` |
| Toolchain | the pinned `nightly-2026-08-28` (`rustc 1.100.0-nightly (e457a7b0d 2026-08-27)`) builds natively; `aarch64-apple-darwin` is tier 1 | measured; https://doc.rust-lang.org/nightly/rustc/platform-support.html : "aarch64-apple-darwin ARM64 macOS (11.0+, Big Sur+)" under "Tier 1 with Host Tools" |

Memory bandwidth and the largest unified memory, by chip class (Apple's figures; "up to" is Apple's wording):

| Class | Bandwidth | Largest memory | Source and copied line |
|---|---|---|---|
| M1 Pro | 200 GB/s | 32 GB | https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/ : "M1 Pro can be configured with up to 32GB of fast unified memory, with up to 200GB/s of memory bandwidth" |
| M1 Max | 400 GB/s | 64 GB | same page: "M1 Max also offers a higher-bandwidth on-chip fabric, and doubles the memory interface compared with M1 Pro for up to 400GB/s" |
| M2 Ultra | 800 GB/s | 192 GB | https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/ : "M2 Ultra features 800GB/s of system memory bandwidth, far greater than anything found in a PC." |
| M4 | 120 GB/s | 32 GB | https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/ : "M4 supports up to 32GB of unified memory and has higher memory bandwidth of 120GB/s." |
| M4 Pro | 273 GB/s | 64 GB | same page: "M4 Pro supports up to 64GB of fast unified memory and 273GB/s of memory bandwidth" |
| M4 Max | 546 GB/s | 128 GB | same page: "M4 Max supports up to 128GB of fast unified memory and up to 546GB/s of memory bandwidth" |
| M5 | 153 GB/s | not stated on the page read | https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ : "a nearly 30 percent increase in unified memory bandwidth to 153GB/s" |
| M5 Pro | 307 GB/s | 64 GB | https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/ : "M5 Pro supports up to 64GB of unified memory with higher unified memory bandwidth up to 307GB/s." |
| M5 Max | 614 GB/s | 128 GB | same page: "M5 Max supports up to 128GB of unified memory with higher unified memory bandwidth up to 614GB/s." |
| **M5 Ultra** | **1.2 TB/s** | **512 GB** | https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/ : "Combined with up to 512GB of unified memory and 1.2TB/s of memory bandwidth, 50 percent higher than before, Mac Studio lets users run massive models entirely on device" |

M5 Ultra GPU: same page, "Its up-to-80-core GPU, the most powerful Apple silicon GPU ever, brings Neural Accelerators to the
Ultra chip for the first time, enabling up to 4.3x the peak AI compute performance when compared to M3 Ultra." The same
page says "Mac Studio with 512GB of unified memory is coming in late October." The M5 Ultra's CPU core count, GPU clock,
peak TFLOPS and the Neural Accelerator's rate per data type are not stated on the pages read: `[unverified]`.

Neural Accelerators (the M5 generation's GPU matrix hardware): https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ :
"The 10-core GPU features a dedicated Neural Accelerator in each core, delivering over 4x peak GPU compute compared to M4,
and over 6x peak GPU compute for AI performance compared to M1." and "Developers can also build solutions for their apps
by directly programming the Neural Accelerators using Tensor APIs in Metal 4." More in section C.4.

## Part A. Native compile census

**What was run.** For each of the 21 non-device crates (the 14 that `python3 tools/recipes.py pure-crates` selects and
the 7 it rejects without a device root), in dependency order, `cargo check -p <crate> --all-targets --keep-going --color never`
with the pinned toolchain's `bin` first on `PATH` and `CARGO_TARGET_DIR` the worktree's `target/`, no `--target`, no env
file: the way `tools/mac-check.sh` runs its native `test` (header `tools/mac-check.sh:9-14`). `--keep-going` is added so one
red target does not hide the others of a crate. One log a crate: `target/mac-portmap/check/<crate>.log` and `.rc`. The 10
device crates (`bloomery-gpu*`, `bloomery-app`) were not built: `AGENTS.md` "Never" forbids plain cargo on them. The
excluded crates (`crates/oxide-ice-unroll`, `fuzz`, `labs/hip-native`, `Cargo.toml:4`) were not looked at.

### A.1 Per crate (layer 1: the tree as it is)

| Crate | rc | Own errors | Distinct causes |
|---|---|---|---|
| bloomery-levers, -jinja, -sampler, -verbs, -models, -refset, -runtime, -tokenizer, -vision, -embed, -placement, -hf, -decision, -serve | 0 | 0 | none (14 crates, all targets: lib, bins, tests, examples) |
| bloomery-gguf | 101 | 1 (lib test only; the lib and every other target build) | `libc::RUSAGE_THREAD` |
| bloomery-threads | 101 | 20: 16 in the lib, 4 more in the lib test | `libc::cpu_set_t`, `CPU_SET`, `sched_setaffinity`, `sched_getaffinity` |
| bloomery-qdot | 101 | 0: stops at `bloomery-threads` | the dependency's 16 |
| bloomery-engram | 101 | 0: stops at `bloomery-threads` | the dependency's 16 |
| bloomery-engram-lab | 101 | 0: stops at `bloomery-threads` | the dependency's 16 |
| bloomery-model | 101 | 0: stops at `bloomery-threads` | the dependency's 16 |
| bloomery-model-hf | 101 | 0: stops at `bloomery-threads` | the dependency's 16 |

Layer 1 totals: 14 crates rc 0, 7 crates rc 101; 21 distinct errors (`gguf` 1, `threads` 20), 2 distinct causes. The five
crates behind `threads` show nothing of their own, so a second and a third layer were measured in a scratch copy of the
tree (outside the worktree, a scratch directory's `layer2` and `layer3`; the copies are not part of
the repository and the worktree is untouched):

- **Layer 2**: `threads`' affinity calls replaced by stubs (a `shim` module returning refusal). Logs
  `target/mac-portmap/check-layer2/`. `qdot` builds, rc 0 on every target (lib, 2 bins, 3 test files). `engram` has 5 errors in the lib,
  `model` 14 in the lib (`engram-lab` and `model-hf` stop at those), and `threads`' own `tests/helper.rs` 4.
- **Layer 3**: also the `libc` items of layer 2 stubbed and the three type mismatches cast. Logs
  `target/mac-portmap/check-layer3/`. `threads`, `qdot`, `engram-lab`, `model` and `model-hf` build on every target (rc 0), including
  `model`'s bins and tests. `engram` has one error left in `tests/engram.rs:885`.

So the host chain's compile surface for the Mac is **45 errors at 38 source lines in 10 files, 8 distinct causes** (the two type-shape breaks counted as one); with
those fixed, every non-device crate builds natively. The stubs were scratch only; `qdot` needed no change.

### A.2 Per site

`lib` is the crate's library; `test` is a `#[cfg(test)]` module or a `tests/` file. "macOS form" is the replacement the
site needs; a form marked `[unverified]` was not tried.

| Site | Crate | Item | What it does | Kind | macOS form |
|---|---|---|---|---|---|
| `crates/threads/src/lib.rs:757` (`mask_cpus`, :750) | threads | `libc::sched_getaffinity` | reads the calling thread's allowed cpus; the pool cuts its topology by it | lib | none: macOS has no affinity mask; the mask is every online cpu (`hw.logicalcpu`) |
| `crates/threads/src/lib.rs:810-823` (`pin`, :809) | threads | `cpu_set_t`, `CPU_SET`, `sched_setaffinity` | pins a pool worker to one cpu; a refusal is returned, not fatal (:806-808) | lib | none: `thread_policy_set(THREAD_AFFINITY_POLICY)` returned `KERN_NOT_SUPPORTED` (46) on this M1 Pro (measured); return `Unsupported`, the pool floats. `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` returned 0 (measured) and is the only placement lever `[unverified]` for P versus E cores |
| `crates/threads/src/helper.rs:217-226` (`float_everywhere`, :216) | threads | same | widens the caller's mask to every cpu | lib | none (no mask to widen): `true` |
| `crates/threads/src/helper.rs:233-242` (`pin_to`, :232) | threads | same | pins the helper thread | lib | as `pin` |
| `crates/threads/src/lib.rs:844-848` (`on_cpus`, :839) | threads | same | a test thread pins itself to read its own mask | test | skip by name on macOS |
| `crates/threads/tests/helper.rs:15-19` (`set_mask`, :10) | threads | same | the helper gate sets the opener's mask | test | skip by name on macOS (the file says "Linux only", :3) |
| `crates/gguf/src/lib.rs:1322` (`thread_faults`, :1318) | gguf | `libc::RUSAGE_THREAD` | counts one thread's page faults for a populate test | test | `getrusage` on macOS takes only `RUSAGE_SELF` and `RUSAGE_CHILDREN` (`man 2 getrusage`, this Mac); a per-thread fault counter was not found `[unverified]`: count process faults in a single-test run |
| `crates/engram/src/direct.rs:39` (`open_direct`, :35) | engram | `libc::O_DIRECT` | opens a file to bypass the page cache | lib | `open`, then `fcntl(fd, F_NOCACHE, 1)` (`man 2 fcntl`: "F_NOCACHE Turns data caching off/on."); the alignment contract (`DIRECT_ALIGN`) is Linux's, whether `F_NOCACHE` needs it `[unverified]` |
| `crates/engram/tests/engram.rs:885` (:871) | engram | `libc::O_DIRECT` | a gate reads `/proc/self/fdinfo` flags | test | skip by name |
| `crates/engram/src/lib.rs:546-550` (`fadvise_dontneed`, :542) | engram | `posix_fadvise`, `POSIX_FADV_DONTNEED` | drops a file range from the page cache after an eviction | lib | no equivalent found; a no-op with a named note `[unverified]`. `F_NOCACHE` affects later reads only |
| `crates/engram/src/lib.rs:702` (`populate`, :701) | engram | `MADV_POPULATE_READ` | faults a row's pages in | lib | `man 2 madvise` on this Mac lists no `MADV_POPULATE`: read one byte a page (the Linux path already has this fallback for kernels older than 5.14, `crates/model/src/placement/host_lock.rs:889-937`) |
| `crates/engram/src/lib.rs:755` (`mincore_chunk`, :749) | engram | `libc::mincore` | counts resident pages | lib | the vector is `*mut c_char` here (`man 2 mincore`: `char *vec`): cast `.cast()`; the bit is `MINCORE_INCORE` |
| `crates/model/src/fileio.rs:29` (`free_bytes`, :19) | model | `statvfs` fields | free bytes on a filesystem | lib | `f_bavail` is `u32` and `f_frsize` `u64` on macOS: widen both |
| `crates/model/src/fileio.rs:37-42` (`rename_noreplace`, :33) | model | `renameat2`, `RENAME_NOREPLACE` | atomic rename that refuses an existing target | lib | `renamex_np(from, to, RENAME_EXCL)` (`man 2 rename`) |
| `crates/model/src/placement/host_lock.rs:867-871` (`drop_pages`, :818) | model | `posix_fadvise`, `POSIX_FADV_DONTNEED` | drops pages of a set's file | lib | as `fadvise_dontneed` |
| `crates/model/src/placement/host_lock.rs:909` (`advise_populate`, :901) | model | `MADV_POPULATE_READ` | faults a span in | lib | as `populate` (the `Populated::Touched` path exists, :889-937) |
| `crates/model/src/placement/host_lock.rs:1218` (`resident_pages`, :1207) | model | `libc::mincore` | resident page count | lib | as `mincore_chunk` |
| `crates/model/src/r8file.rs:512-517` (`sync_and_drop`, :510) | model | `sync_file_range`, `SYNC_FILE_RANGE_*` | writes a range out before dropping it | lib | `fcntl(F_FULLFSYNC)` after the writes (`man 2 fcntl`: "F_FULLFSYNC Does the same thing as fsync(2) then asks the …") |
| `crates/model/src/r8file.rs:528` (`drop_cached`, :525) | model | `posix_fadvise`, `POSIX_FADV_DONTNEED` | drops written pages | lib | as `fadvise_dontneed` |

Sites that compile on a Mac and still do the wrong thing there are not compile errors, so the census cannot see them.
Two kinds were listed by reading the code:

- **Linux files** (`/proc`, `/sys`): 40 lines in 12 files (a grep, `target/mac-portmap/proc-sys-sites.txt`; a few are message strings or `#[ignore]` reasons, not reads). In the
  library and the load path: `crates/threads/src/topology.rs:83-129` and `helper.rs:177` (sysfs topology; with no sysfs
  `detect_topology` returns one group of one cpu, `topology.rs:99-112`, so the pool's default is one thread `[derived]`);
  `crates/model/src/placement/workstation.rs:35-90` (`/proc/meminfo`, `/proc/self/cgroup`, `/sys/fs/cgroup`: the host room).
  In tools and tests only: `crates/model/src/bin/bench_v41_host.rs` (11 lines), `crates/engram-lab/src/bin/engram-rate.rs`
  (6) and `src/lib.rs:51-58`, `crates/model/src/bin/bloomery-decode.rs:143`, `crates/model/src/ops.rs:6573`,
  `crates/serve/tests/limits.rs:26`, `crates/threads/tests/pool.rs:208,351`.
- **Units and sizes**: `ru_maxrss` is KB on Linux and bytes on macOS (`man 2 getrusage`: "the maximum resident set size
  utilized (in bytes)"), read at `crates/model-hf/src/bin/fixture.rs:387` and `crates/model-hf/tests/fixture.rs:852`.
  `ROW_PAGE` is 4096 (`crates/placement/src/placement/workstation.rs:141`) and `host_available_read` refuses any other page
  size before it reads anything, even with `BLOOMERY_HOST_ROOM` set (`crates/model/src/placement/workstation.rs:24-28`); this Mac's
  page is 16384 (measured), so every placement plan on a Mac stops there until the page becomes a parameter. The row-room
  arithmetic uses the constant too (`crates/placement/src/placement.rs:3055-3062`). The cache line is 128 B here (measured);
  code that pads to 64 B for false sharing still works.

### A.3 Against `recipes.py pure-crates` and the Spark port

- `tools/recipes.py pure-crates` rejects exactly the 17 crates found: 10 for the device root and the 7 above, each with its
  first Linux-only item. It rejects `bloomery-gguf` for `RUSAGE_THREAD` at `gguf/src/lib.rs:1322`; the census agrees that the
  failure is in the lib test alone (`cargo test -p bloomery-gguf` does not build, the lib does).
- The rule's name list (`tools/recipes.py:5512-5525`) does not hold `libc::O_DIRECT`, and no name list can hold the two
  type-shape breaks (the `mincore` vector, the `statvfs` field widths). Each sits in a crate the rule already rejects, so
  today nothing is hidden; a crate fenced by name would still fail natively.
- The rule's `#[target_feature(` pattern (`tools/recipes.py:5514`) does not look at the feature: a NEON body under
  `cfg(target_arch = "aarch64")` is true on the Mac and would reject `qdot` again (Part D).
- The Spark port fenced every x86 site (`dgx-spark.md` Appendix A): `qdot` under `cfg(target_arch =
  "x86_64")` (`crates/qdot/src/lib.rs:62`), the prefetch in `crates/model/src/ops.rs:5875-5893`, `deepseek2/attn.rs`, the bin and
  the tests; those hold on the Mac (the census shows no x86 error). The Spark port left the libc items out because they are
  POSIX on Linux ("not blockers on DGX OS `[unverified until a compile]`", `dgx-spark.md` Appendix A). They are the whole
  of the Mac's census: 45 errors, none of them an x86 item.

## Part C. The Metal backend (M2), design

The decode step on this engine is bandwidth-bound on its weights and launch-bound on its chain; prefill is compute-bound.
The design below is cut to that: each choice states what it costs or gains, `[derived]` where it is computed. Where
the number needs a Mac measurement that cannot be derived, the term is named and left open (section C.5).

### C.1 The device seam

**What `crates/gpu` exposes, by kind** (the callers are the family crates `gpu-deepseek41`, `-glm5next`, `-mimo2`, `-solar2`,
`-vision`, `-gemma4a`, `-gemma-embedding2`, `gpu-gates` and `app`; counts are greps over those crates and `crates/gpu`, the
command is in `target/mac-portmap/surf.sh`; they are `[derived from code]`):

| Kind | What the code uses | Count | Examples |
|---|---|---|---|
| Device memory | `cuda_core::DeviceBuffer<T>`: `zeroed` 449, `from_host` 201; typed windows | 280 import lines | `crates/gpu/src/arch/qwen3moe/scratch.rs:90`, `crates/gpu/src/tensor.rs:41` (`window`), `crates/gpu/src/weights.rs:54` (`DevWeight`) |
| Pinned and mapped host memory | `PinnedHostBuffer` 16 sites; `cuMemHostAlloc` 22; `cuMemHostGetDevicePointer` 8; a mapped page shared with a tier node | 46 | `crates/gpu/src/checkpoint.rs:330`, `crates/gpu/src/graph.rs:491` (`MappedHost`), :544 |
| Copies and fills | `cuMemcpyHtoDAsync` 18, `DtoH` 10, `DtoD` 4, `Memcpy2D` 8, `cuMemsetD32Async` 4 | 44 | `crates/gpu/src/graph.rs` (`TapRows` copies are graph nodes) |
| Streams | `CudaStream` (the stream every launch takes), per-role streams | 250 import lines | `crates/gpu/src/lib.rs:2281` (`StreamRole`), :2290 (`role_stream`) |
| Events, host sync | `CudaEvent`, `.record` 37, `.wait` 37, `.synchronize()` 107, `cuLaunchHostFunc` 8, `cuStreamBatchMemOp` 8 | | `crates/gpu/src/checkpoint.rs:93`, `graph.rs:341` (`Branch`/`Forked`), :781-824 (`op_wait_geq`, `op_write`), :882 (`HostFlags`) |
| Graph capture and replay | `Gpu::capture`, `Graph::launch`, node introspection (`cuGraphGetNodes` 5, `cuGraphLaunch` 4) | | `crates/gpu/src/lib.rs:2776`, `crates/gpu/src/graph.rs:51,82,186,204` |
| Kernel launch | `#[cuda_module]` 90 blocks, `pub fn enqueue_*` 316, `prepare_*` per kernel; arguments are typed buffers and `u32`s | 316 | `crates/gpu/src/lib.rs:3021-3052` (`q4k_gemv_prefetch`/`q4k_gemv`/`q4k_gemv_mcol`) |
| Module loading | embedded device module loaded once per context | | `crates/gpu/src/lib.rs:2425` (`bundle_module`) |
| Fault word | one `u32` array per `Gpu`, kernels raise by atomic min | | `crates/gpu/src/fault.rs` (`ptx_asm!`), `crates/gpu/src/lib.rs:2549-2617` |
| Device query, memory room | `census`, `mem_info`, `allocation_granularity`, integrated flag | | `crates/gpu/src/lib.rs:2146`, :2692, :2705 |

Device-side, the 376 `#[kernel]` fns use what NVIDIA exposes: `cuda_device::wmma` (`mma_m16n8k16` 22, `ldmatrix` 14),
`dotprod::dp4a_s32` (`crates/gpu/src/cores.rs:14,224`, `quant_sel.rs`, `kquant/walk.rs`, `q5.rs`, `head_argmax.rs`),
`async_copy` 8, `warp::shuffle_xor_f32`, `float::{fma_rn,mul_rn,add_rn}_f32` and `ptx_asm!` in 7 places in 5 files
(`fault.rs`, `cores.rs`, `flash_gqa_prefill.rs`, `linear/delta.rs`, `gemm/gemm32.rs`). Metal has no `dp4a`: `dot(char4,char4)`
does not compile, and `simdgroup_matrix` takes `half`, `bfloat` and `float` only (both measured on this M1 Pro,
`target/mac-portmap/msl-probe.log`; the spec line is https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf , MSL
4.1, section 2.4: "Metal supports the following SIMD-group matrix type names, where T is half, bfloat (in Metal 3.1 and later) or
float and Cols and Rows are 8"). An atomic min on a device `atomic_uint` compiles (measured), so the fault word moves.

**The proposal.** Two cuts, in this order.

1. **A device layer, one owner, selected at build time.** A new crate, `bloomery-device`, owns exactly the kinds above that
   are resource and scheduling types: `Buffer<T>`, `Pinned<T>`, `Stream`, `Event`, `Graph` (capture, replay, node count),
   `HostFlags` and `MappedHost`, the fault word, the device census. Its public API names no `cuda_core` and no Metal type.
   The CUDA implementation wraps today's types without changing their behaviour; the Metal one is new. The family crates
   import the layer's names instead of `cuda_core`'s (about 280 import lines, a rename). It is selected by a Cargo feature,
   not by a trait object: the hot path is 12 L + 4 launches a token (C.2), and an indirect call per launch is a cost
   with no gain, since a build holds one backend. This follows `AGENTS.md` "common code first": the owner of the resource
   types is one place; no common owner grows a branch per backend, because the branch is the module boundary.
2. **The chain walk stays common; the launches go behind a small op set.** `crates/gpu/src/arch/qwen3moe/dispatch.rs` mixes the
   order of the chain (`layer` :422, `mixer` :439, `ffn` :989: what runs after what, from the `LayerPlan`) with the launches
   (which kernel an op takes by weight type). The order and the plans (`plan.rs`, `program.rs`, the arena sizing in
   `scratch.rs`) do not name a backend except through `DeviceBuffer`. The launches are the 15 ops of C.2. The cut is one
   trait of those ops, `ChainOps`, with an associated buffer type from the device layer, implemented by the CUDA kernels
   (the existing wrappers, moved) and by the Metal ones; the walk is generic over it. A backend refuses an (op, weight
   type) pair it does not implement by a named error, as `GateUpEntry`/`DownEntry` do today
   (`crates/gpu/src/arch/qwen3moe/dispatch.rs:1093-1230`). The 316 `enqueue_*` wrappers of other families stay as they are:
   they move to the op set only when a second family reaches Metal. The alternative, a Metal copy of the chain
   (`dispatch.rs` is 1513 lines), is the per-model copy `AGENTS.md` allows only for a large measured speed gain; no such
   gain is known, so the copy is not proposed. Part F asks the maintainer to confirm the cut.

The serve side already has a seam of this size: `trait Seat` (`crates/gpu-gates/src/bind.rs:1268-1515`, 12 required methods)
is what `SeatEngine` (:1519, the only `serve::Engine` over a real model, impl at :2983) drives. `bind.rs` names 21 lines of
`gpu`/`cuda`/`nvidia` items and its `Vocab` (:81, the `serve::Tokenizer`) is in the device-linked crate `gpu-gates`. For the
first Metal decode a CLI that feeds ids needs none of it; for `bloomery-serve` on a Mac they lift into a device-free crate (section B.2).

**Fault, snapshots, checkpoints.** The fault word moves as is (atomic min on a shared `atomic_uint`, read back with the
token). `Seat::snapshot`/`resume` (`bind.rs:1445,1465`) and `crates/gpu/src/checkpoint*` use CUDA events and pinned staging;
the Metal engine's first version answers them with a named refusal and the serve gates that need them are deferred by name.

### C.2 The Qwen3-MoE decode kernel set `[derived from code]`

Derived from `crates/gpu-gates/src/bin/generate_qwen3moe.rs` (the decode CLI, header :1-60), which opens a model through
`Qwen3moeModel` and runs `dispatch::enqueue_chain` (`crates/gpu/src/arch/qwen3moe/dispatch.rs:96`, a `Program` walk at
`m` = 1). The case read: a `qwen3moe` file (no shared expert, head 128 NeoX attention), whole on the card, f16 K/V, the
common Q4_K/Q6_K mix. A launch per layer is `pass_launches` (`dispatch.rs:303-382`): 1 (`norm_quant`) + 1 (fused q/k/v)
+ 3 (K/V append, flash segment, flash merge) + 2 (activation quantizer, fused output projection with the residual) +
5 (router, gate·up with SwiGLU, quantizer, down, combine) = **12 launches a layer**, plus 1 per layer whose V is Q6_K,
plus the front (1) and the head (3): **12 L + 4 launches a token** for L layers. Each kernel below has a `_prefetch` twin
for `Gpu::row_prefetch()` (`crates/gpu/src/lib.rs:2673`), chosen at launch; the table lists the base.

| # | Kernel | Where | What it computes | Quant types it takes |
|---|---|---|---|---|
| 1 | `embed_rows_q4k` | `crates/gpu/src/elem.rs:682` | token row → `x`; also writes position and live key count | Q4_K; siblings `_q6k` :898, `_q5k` :849, `_q3k` :946, `_q5_0` :760; Q8_0 through `q38` (`dispatch.rs:504-519`) |
| 2 | `norm_quant` | `crates/gpu/src/fused.rs:99` | RMSNorm of `x` and the q8_1 quantization of the result | f32 → q8_1 |
| 3 | `qwen3moe_qkv_q4k` | `crates/gpu/src/arch/qwen3moe/proj.rs:56` | fused Q4_K q/k/v gemv against the q8_1 activation | Q4_K; V in Q6_K: `q6k_gemv` (`lib.rs:1340`); other: `q4k_gemv` (`lib.rs:1086`) |
| 4 | `head_norm_neox_append` | `crates/gpu/src/rope_neox.rs:875` | per-head Q/K RMSNorm, NeoX rope, K/V row appended to the cache | f16 K/V; Q8 cache: `_q8` :1265 |
| 5 | `gqa_flash_seg_mma` (scalar twin `gqa_flash_seg`) | `crates/gpu/src/flash_gqa.rs:3230` (:2956) | split-K flash attention segment over the live keys | f16 K/V |
| 6 | `gqa_flash_merge` | `crates/gpu/src/flash_gqa.rs:3477` | merges the segments' partial softmax | f32 |
| 7 | `q3k_quantize_q8_1` | `crates/gpu/src/lib.rs:912` | q8_1 quantization of the attention rows (the same kernel quantizes the SwiGLU rows at 11, and the head's input) | f32 → q8_1 |
| 8 | `qwen3moe_o_resid_q4k` | `crates/gpu/src/arch/qwen3moe/proj.rs:355` | Q4_K output projection plus the residual add | Q4_K |
| 9 | `softmax_router_norm` | `crates/gpu/src/route_softmax.rs:755` | FFN RMSNorm and q8_1 (every block), the router logits, then the ticketed routing (`publish_route`); one launch at `m` = 1 | f32 router weights |
| 10 | `qwen3moe_gate_up_swiglu_q4k` | `crates/gpu/src/arch/qwen3moe/experts.rs:136` | gate and up of the chosen experts with SwiGLU | Q4_K (other types: `elem::swiglu` :1137 and the `_sel` family: Q5_K, Q8_0, IQ2_XS/IQ3_XXS/IQ3_S/IQ4_XS, MXFP4, NVFP4, `dispatch.rs:1093-1230`) |
| 11 | (7 again) | | quantizes the SwiGLU rows | |
| 12 | `q4k_gemv_sel` / `q6k_gemv_sel` | `crates/gpu/src/q4k_sel.rs:344`, `q6k_sel.rs:166` | down projection of the chosen experts | Q4_K / Q6_K (also Q5_K, Q5_0, Q5_1, Q8_0, IQ*, MXFP4, NVFP4) |
| 13 | `qwen3moe_combine` | `crates/gpu/src/arch/qwen3moe/experts.rs:280` | weighted sum of the experts, plus the residual | f32 |
| 14 | `rms_norm` | `crates/gpu/src/elem.rs:1004` | the head's final RMSNorm | f32 |
| 15 | `qwen3moe_head_q6k_argmax` | `crates/gpu/src/arch/qwen3moe/head_argmax.rs:80` | Q6_K output projection with the argmax folded in | Q6_K (another head type: the shared head's gemv and `argmax_fault` :1410) |

**Count: 15 distinct kernels** for a Q4_K/Q6_K file (16 with a Q6_K V), 7 of them (`qkv`, `o_resid`, `router_norm`, `gate_up_swiglu`, both `_sel`, the head) with a `_prefetch` twin, `[derived from
code]`. The family also holds the prompt-side set (`flash_gqa_prefill`, the grouped GEMM of `wide.rs`, `gemm/`) that the
decode does not launch; a file of other quant types adds the `_sel` variants named in the rows. The entry bodies of these 15
total 1436 lines `[derived]` (summed `#[kernel]` extents of the rows above, scalar flash counted once); the device cores they
call (`cores.rs` 1455 lines, `quant_sel.rs` 1009, `route_body.rs` 806, `route_core.rs` 106) are shared with other kernels,
so the port is about 1.4k lines of entries plus the part of 3.4k lines of cores these reach, `[derived]` order of magnitude.
Qwen3.6 (`qwen35moe`, the fixture family, `crates/model/src/arch/registry.rs:124`) runs the same `dispatch::layer` and adds the
delta rule's three kernels, the gated router and the gated quantizer (`body.rs:68-76`, "The kernels only Qwen3.6's layers launch"). The `qwen3moe` registry row has no fixture (`registry.rs:100-104`).

### C.3 Kernel authoring for Metal

Six options, each judged on: exists and maintained, one source for CUDA and Metal, cost per kernel, risk to the ggml
bit band (`AGENTS.md`: kernels match ggml within each crate's RESULTS band), and the CUDA-graph equivalent. The Metal
equivalent of a CUDA graph for all of them is an indirect command buffer: https://developer.apple.com/tutorials/data/documentation/metal/mtlindirectcomputecommand.json ,
"A compute command in an indirect command buffer."; https://developer.apple.com/tutorials/data/documentation/metal/indirect-command-encoding.json , "Store draw
commands in Metal buffers and run them at a later time on the GPU, either once or repeatedly." The contract matches the
graph's: fixed buffers in place, fixed grids, the per-replay values in a device record (`StepParams`,
`crates/gpu/src/arch/qwen3moe/body.rs:1000-1003`; the graph's rule at `crates/gpu/src/graph.rs:47-50`). Whether Metal 4 keeps the ICB
as the replay path for compute is in the pages above; the cost per replayed dispatch is a Mac measurement (C.5).

| Option | Exists, maintained | One source for CUDA and Metal | Cost per kernel | Risk to the bit band | Tensor ops / graph |
|---|---|---|---|---|---|
| **Hand-written MSL** | MSL 4.1 spec dated 2026-06-04 (the PDF above) | no: a second source per kernel; the CUDA kernels stay | the 15 decode kernels are 1.4k entry lines in Rust (C.2); MSL is the same order. A kernel's loop order is copied, so its bits are controlled | lowest: `#pragma METAL fp math_mode(safe)` and `fp contract(off)` compile (measured). The spec says "The default is fast." and "By default, the compiler allows floating-point contractions. For example, a*b+c may be converted to a single fused-multiply-add." (MSL 4.1, the compiler options, pages 15-16 of the PDF) | full: `simdgroup_matrix`, MPP `matmul2d`, ICB, `MTLSharedEvent` |
| **Rust to SPIR-V (rust-gpu) to MSL (SPIRV-Cross)** | https://github.com/Rust-GPU/rust-gpu pushed 2026-10-16: "Currently only [SPIR-V] support is planned" (the README's own link markup); https://github.com/KhronosGroup/SPIRV-Cross pushed 2026-10-14: "Convert SPIR-V to readable, usable and efficient Metal Shading Language (MSL)" | no: kernels are rewritten against `spirv_std`, not `cuda_device` | the rewrite of a hand-written MSL, plus a two-step compile | high: SPIRV-Cross output is compiled by Apple's compiler at its default math mode; no way to name `simdgroup_matrix` or MPP from SPIR-V | none for tensor ops; ICB by the host |
| **CubeCL** | https://github.com/tracel-ai/cubecl pushed 2026-10-20: "A single `#[cube]` Rust function is compiled on demand and runs on CUDA, HIP, Metal, Vulkan, WebGPU, or CPU"; https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-metal README: "Metal 3+ required (atomic float and simdgroup matrix support)" | yes, but not with cuda-oxide: its own `#[cube]` DSL and codegen replace the 376 kernels' language | a rewrite per kernel into the DSL | high: the codegen's op order and contraction are CubeCL's; Metal tensor ops: not found in the crate list or README `[unverified]` | its own runtime, not ICB `[unverified]` |
| **Slang** | https://github.com/shader-slang/slang pushed 2026-10-21: Metal "experimental\*"; "Slang currently supports generating vertex, fragment, compute, task and mesh shaders for Metal." | yes, in a new language (CUDA output is C++) | a rewrite per kernel in Slang | high, as SPIRV-Cross | none known `[unverified]` |
| **MLX kernels as a library** | https://github.com/ml-explore/mlx pushed 2026-10-20 (MIT): "MLX is an array framework for machine learning on Apple silicon"; kernels in `mlx/backend/metal/kernels/` include `*_nax` files | no | none to write; but `quantized.h` is group-affine quantization (`group_size`, 0 hits for `q4_K`/`gguf`/`block_q`): GGUF K-quants are not served, so a model would be requantized | n/a: a different numeric (scales and biases per group) | uses TensorOps for the Neural Accelerators (https://machinelearning.apple.com/research/exploring-llms-mlx-m5 : "MLX leverages the Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4 to support the Neural Accelerators’ features.") |
| **ggml-metal kernels as a library** | https://github.com/ggml-org/llama.cpp pushed 2026-10-21 (MIT): "Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks"; local checkout `36a73916e`: 978 `host_name` entries in 14841 lines of `ggml/src/ggml-metal/kernels/*.metal,.h` | no | none to write for 30+ quant types: `kernel_mul_mv_id_q4_K_f32` and the rest (`kernels/mul_mv.metal:3362` and below); the host side (`ggml-metal-ops.cpp`) is tied to the ggml graph and would be rewritten | different numerics: the K-quant mat-vec reads f32 activations (`kernels/mul_mv.metal:1546`, `float * y`), not q8_1; so its bits are not the card engine's, and their band against ik is not measured `[unverified]` | tensor API behind `GGML_METAL_HAS_TENSOR` (`kernels/mul_mm.metal:13`); ggml's own comment: "disable the tensor API by default for old chips because with the current implementation it is not useful - M2 Ultra: ~5% slower - M4, M4 Max: no significant difference" (`ggml-metal-device.m:1151-1153`) |

**Recommendation: hand-written MSL, one kernel at a time, structured after ggml-metal's where the numerics allow.**
Reasons, each against the goal (speed on the M5 generation):

- The two things that decide speed on M5, the Neural Accelerators and the dispatch chain, are reached only by MSL written
  against MPP `matmul2d` and by an ICB the engine encodes itself. A cross-language route (SPIR-V, Slang) has no door to
  either; CubeCL has none found.
- The numeric contract is ours to hold. A kernel that must match a CUDA kernel bit for bit (the integer-exact ones, C.4) can
  only be written with the loop order and the fused-multiply-add placement under our control, which the compile flags
  above give and a generator does not.
- ggml-metal is the structure to learn from and the oracle for what is fast on Apple silicon; it is MIT, so a kernel adapted
  from it carries its notice (`THIRD_PARTY_NOTICES.md`). It is not the runtime: its activations are f32, so using it leaves
  the engine's q8_1 path and makes Metal's results a different family from the CUDA engine's.
- Cost: the second source per kernel. It is bounded: 15 kernels for the first model (C.2), each gated against a reference
  (Part D). The decision between "q8_1 activations as on CUDA" and "f32 or f16 activations as ggml-metal" is the first open
  question (Part F), since it sets what the first kernels are.

### C.4 Designing for the M5 Ultra

**What M5 adds, and how Metal exposes it.** A Neural Accelerator in each GPU core, programmed through Metal 4 tensor APIs
(section 2). On the shader side that is `<metal_tensor>` and `MetalPerformancePrimitives`, in the SDK on this Mac:
`MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h` (macOS 27.0 SDK, copied to
`target/mac-portmap/MPPTensorOpsMatMul2d.h.txt`). Its header lists the data type combinations of `matmul2d` (left, right,
destination), among them `half half half`, `half int8_t half`, `int8_t int8_t int32_t`, `uint8_t uint8_t int32_t`, `bfloat bfloat float`,
`int8_t int4b_format int32_t`, `int8_t int2b_format int32_t`, `half metal_fp4_e2m1_format half` and the fp8 formats, and a descriptor
flag `relaxed_precision` ("set it to true to allow implementation to sacrifice accurancy for performance", sic). The host
side is `MTLTensor` and the `MTL4*` encoders (https://developer.apple.com/tutorials/data/documentation/metal/mtltensor.json , "A resource representing a
multi-dimensional array that you can use with machine learning workloads."; the shader can also build a `tensor_inline` from a device
pointer, as the header's example and the probe below do).

**Measured on the M1 Pro** (`target/mac-portmap/mpp-int8-probe.log`, `mpp-types-probe.log`): an `matmul2d` of `int8_t × int8_t → int32_t`
(64 × 32 tile, 4 simdgroups, `tensor_inline` views of device buffers) compiles and builds a pipeline on an `apple7` GPU, runs,
and equals the CPU's integer product on all 8192 outputs (M = 128, N = 64, K = 96); `half×half→float`, `bfloat×bfloat→float`,
`float×float→float`, `half×int8_t→float` and `int8_t×half→float` also build. So the tensor-op path can be written and checked
bit for bit on the M1 dev machine. Whether it is fast there is not what the check shows; ggml's comment above says the tensor API is not
faster than `simdgroup_matrix` on M2 Ultra and M4. Whether the M5's Neural Accelerators run `int8` at a higher rate than `half`, and the
rate itself, is not stated on any page read: `[unverified]`. Tuning the matrix kernels therefore needs an M5-class machine (Part F).

Apple's own description of where the matrix hardware pays (https://machinelearning.apple.com/research/exploring-llms-mlx-m5): "In LLM
inference, generating the first token is compute-bound, and takes full advantage of the Neural Accelerators." and "Generating
subsequent tokens is bounded by memory bandwidth, rather than by compute ability." That is this engine's split, so:

| Choice | What it is | Speed: cost or gain |
|---|---|---|
| **Decode streams weights at bandwidth** | one-row (GEMV) kernels, no matrix hardware; the Q4_K block is read once, decoded in registers | the floor is bytes over bandwidth. For Qwen3.6-35B-A3B `Q4_K_M` the weight bytes a token are 1.8 GB (`dgx-spark.md` section 5, from 3 B active parameters at 4.85 bits `[derived]`), so the ceiling is 1.8 GB / 1.2 TB/s = **1.5 ms a token on an M5 Ultra** `[derived]`, against 9.0 ms on this M1 Pro (200 GB/s), 3.3 ms on an M4 Max (546), 2.9 ms on an M5 Max (614). Datasheet figures; the achievable fraction is a measurement (C.5) |
| **Int8 dot without `dp4a`** | the q8_1 path's integer dot on the ALUs: 4 `int` multiply-adds a `dp4a`; or fp32 FMAs of the same products | exact either way. fp32 holds integers below 2^24 exactly; a 32-value q8 partial sum is at most 32 × 127 × 127 = 516128 `[derived]`, so an fp32 FMA chain per 32-value block gives the same integer as `dp4a`, and the scales come after. At 1.2 TB/s a Q4_K weight stream (256 values in 144 B, 1.78 values per byte `[derived]`) needs 2.1 × 10^12 multiply-adds a second `[derived]`: peak GPU throughput is not stated on the pages read (`[unverified]`), so whether decode stays memory-bound on M5 Ultra is the second open measurement |
| **Prefill and multi-row verify on tensor ops** | the prompt GEMM, the grouped expert GEMM and the draft verify rows (`m` up to `GEMV_COLS`, `crates/gpu/src/arch/qwen3moe/wide.rs`) as `matmul2d` over weight tiles decoded to `half` (or `int8` with the scales after) in threadgroup memory | the CUDA engine already does this with `mma_m16n8k16` over dequantized tiles (`gemm/kernels.rs`, `grouped.rs`); the Metal kernel is the same shape. Apple: "over 4x the peak GPU compute for AI" for M5 over M4. Prompt rows are compute-bound once the weights are read once per ubatch: a ubatch of P rows costs 2 × 3 B × P FLOP against one pass over the 21.2 GB file `[derived: dgx-spark.md section 5]`, so the break-even P is `F_peak × 17.7 ms / 6 GFLOP` with the file read at 1.2 TB/s: for every 10 TFLOPS of achieved matrix rate it is about 30 rows `[derived]`. `F_peak` is `[unverified]` |
| **Chain structure: fewer, fatter dispatches** | the decode is 12 L + 4 launches (C.2). At 1.5 ms of weight time, a gap of g µs between dispatches costs (12 L + 4) g: for L = 48 `[unverified input: the layer count was not read from a file in this round]` that is 580 g µs, so g must stay under 2.6 µs `[derived]` for the chain not to dominate | the term the CUDA engine measured as `c_node` (`AGENTS.md`, "launch count only": Δt = ΔN × c_node). On Metal it is not known. Levers, in the order the CUDA engine used: fuse where it is free (the engine already fuses qkv, o+residual, router+norm, gate·up+SwiGLU), and issue independent dispatches under `MTLDispatchTypeConcurrent` with barriers only between dependent ones. An ICB is not a lever: M2.0 measured it no faster than the plain encoder (below). A fold costs the repeated work times the blocks (`AGENTS.md` "fold a launch into a neighbour") |
| **Weights streamed in place** | wrap the mmap'd GGUF in `MTLBuffer`s without a copy (`newBufferWithBytesNoCopy`; https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:).json , "Creates a buffer that wraps an existing contiguous memory allocation."), as ggml-metal does (`ggml-metal-device.m:2143,2212`) | removes the load copy (the file's bytes are the card's bytes). Costs: page alignment (16384 B here), `maxBufferLength` 20100448256 B on this machine, so a 21 GB file is several buffers (a tensor never spans two; an expert stack per layer is one) |
| **Residency** | keep the weights in an `MTLResidencySet` (https://developer.apple.com/tutorials/data/documentation/metal/mtlresidencyset.json , "A collection of resource allocations that can move in and out of resident memory.") | avoids a first-touch fault inside a replay; the pool can still page under pressure, so the wired limit matters (C.6) |
| **Numeric contract** | integer-exact kernels (the q8_1 dots, the quantizers, the combine's fixed order) bit-equal to the CUDA engine; transcendental ones (`exp` in SwiGLU and softmax, `tanh`) within a band | the integer sums are exact on any hardware; the float closes are IEEE fp32 with the same operation order, so they are equal when the contraction is pinned (`fp contract` and `math_mode(safe)`). `exp` differs in ulps between libdevice and Metal: a band, held by the same gates as the CUDA ones (`AGENTS.md`: accuracy measures are diagnostics, the pins are the bug catchers) |

**Unified memory beyond the plan (section C.6).** The host tier on a unified machine shares DRAM bandwidth with the GPU: on
a Mac an expert computed by the CPU does not add a second memory, it competes for the one. So the CPU tier has no capacity
use on a Mac whose memory holds the model (an M5 Ultra's 512 GB, or this M1 Pro's 26.8 GB working set for the 21.2 GB
Qwen3.6 file), and only the NVMe tier past that. This is why M1 leaves the critical path.

### C.5 Apple Silicon facts, the terms, and the resource timeline

The chip facts the design needs (bandwidth by class, the largest memory, the M5 Ultra's GPU) are in section 2, the matrix hardware in C.4,
the measured device limits in section 2 and C.6. This section holds the terms.

Per decode token, whole model on the GPU, weights resident (`[derived]`; `n/a` is a resource the machine has none of):

| Resource | Demand a token | Note |
|---|---|---|
| GPU memory traffic | 1.8 GB (Qwen3.6-35B-A3B `Q4_K_M`) plus K/V reads at the depth | 1.5 ms at the M5 Ultra datasheet 1.2 TB/s; the K/V term is `2 × layers × kv_heads × head × depth × bytes` and not computed here (hparams not read) |
| GPU launch chain | 12 L + 4 dispatches | M4 Pro (M2.0): 0.84 µs an empty dependent dispatch, 2.08 µs a bandwidth-bound one in the 580-dispatch chain |
| GPU ALU | 2.1 × 10^12 multiply-adds a second to keep pace with the stream at 1.2 TB/s | peak not stated: `[unverified]` |
| Host CPU | encode the step's dispatches into one command buffer, wait on it | under the GPU's time when encoding runs ahead of the GPU `[unverified on Metal]` |
| PCIe | n/a | one memory |
| NVMe | 0 after load | |
| Host DRAM | the same DRAM as the GPU's | a CPU tier steals from the GPU's bytes |

The wall is not a sum: the host's two items run under the GPU's. It is the GPU's bytes plus its chain. A lever on the host CPU
predicts 0; a lever on the chain predicts up to the gap budget above; a lever on the bytes (a smaller weight format, drafted
rows that share one pass over the weights) predicts in proportion. The step is latency, not bandwidth, once the chain term passes the
byte term, and then the busiest unit is not the bound (`AGENTS.md`, the q3gemma case). The two terms a derivation cannot fix are
**c_d**, the cost of one replayed dispatch in an ICB on M-series GPUs, and **f_bw**, the fraction of the datasheet bandwidth a
streaming-read kernel reaches. Both need a Mac runner (Part D) and the first Metal round: an empty-kernel ICB of 580 dispatches
replayed N times, and a read-only kernel over a 2 GiB buffer. The expected values cannot be written from anything measured here;
that is the reason the first round's step 0 is these two measurements, taken before any kernel is ported.

**Measured (M2.0, `tools/metal-probe`, card `docs/cards/mac-m20-dispatch.card`; development evidence, no Mac timing
runner yet).** On an M4 Pro (20-core GPU, 273 GB/s):
- `c_d` = 0.84 µs plain, 0.90 µs ICB and 1.80 µs with one command buffer per dispatch. The card's band was 6.0–8.5 µs.
- `f_bw` = 0.96 alone (262 GB/s).
- The 580-dispatch, 1.8 GB chain takes 8.31 ms in both modes, against 7.10 ms at the in-chain bandwidth (ratio 1.17).

On an M1 Pro: `c_d` = 2.08 µs plain and 1.46 µs ICB (+121 µs a replay), `f_bw` = 0.88, and the chain ratio is 1.14.

What the M4 Pro numbers settle:
- The fixed cost the published engines pay is not per-dispatch encoding.
- The ICB is dropped; the chain encodes plainly.
- The chain's extra ~1.2 µs a dispatch over an empty one is the drain and refill of a bandwidth-bound kernel at each
  barrier `[derived]`. The levers are fewer dependent steps and independent work beside each barrier.

### C.6 Unified memory

What of the Spark's placement applies (`crates/placement/src/placement.rs:280-289` `UnifiedPool`; `:3956-3960` `pool_card_bytes` set when
`machine.unified()` is some; `Violation::PoolOver` :1580, :4182; `tierengine.md`, section (h), line 264):

| Spark (GB10) | On a Mac |
|---|---|
| one pool, `PoolOver` holds Σ card + host ≤ available | the same rule. The census fills `UnifiedPool` from `MTLDevice.hasUnifiedMemory` (https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/hasunifiedmemory.json : "A Boolean value that indicates whether the GPU shares all of its memory with the CPU.") |
| the pool's room is `MemAvailable` | the GPU's share is capped lower: `recommendedMaxWorkingSetSize` ("An approximation of how much memory, in bytes, this GPU device can allocate without affecting its runtime performance.", https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/recommendedmaxworkingsetsize.json) was 26800603136 B of 34359738368 on this machine (0.78 [derived]; the same ratio on three Macs, section 2). `iogpu.wired_limit_mb` read 0 here (the default); a third party's reading of the default is two thirds of RAM up to 36 GB and three quarters above (`mistral.rs` `mistralrs-core/src/utils/memory_usage.rs:398-406`, `[unverified]` for Apple's behaviour). The pool is `min(working set, MemAvailable)` |
| `cuMemHostAlloc(PORTABLE\|DEVICEMAP\|WRITECOMBINED)` pinned arena | `MTLResourceStorageModeShared`: https://developer.apple.com/tutorials/data/documentation/metal/mtlstoragemode/shared.json , "The CPU and GPU share access to the resource, allocated in system memory." and "This is the default storage mode for … instances on integrated GPUs and both … and … instances on Apple silicon GPUs." (the type names are links in the page data) No flags, no registration; write-combined does not exist |
| `cudaHostRegister` of an anonymous arena | not needed: wrap page-aligned memory with `newBufferWithBytesNoCopy` |
| the card leg reads the arena in place | the same: a Metal kernel reads the shared buffer. The CPU and GPU must not touch a buffer at once (Apple: "you’re responsible for synchronizing access to the texture from the other participant"): the `HostFlags` role is `MTLSharedEvent` waits and signals (https://developer.apple.com/tutorials/data/documentation/metal/mtlsharedevent.json : "A shared event can synchronize memory operations across multiple Metal devices and the CPU.") |
| `O_DIRECT` reads land in the arena | `F_NOCACHE` and `pread` into the buffer's pages |
| `ROW_PAGE` 4096 | 16384: the page is a parameter |

## Part B. The CPU path (cut to what M0 and M2 need)

**What a CPU path is for on a Mac.** Three things. The development loop (the host chain builds and the pure crates' tests
run on the Mac today: 1477 passed). The oracle: `qdot`'s scalar mirrors are the definition the card kernels and the
reference sets are held to, and they run on a Mac as they are. And a host tier for experts that do not fit the GPU's
working set; on a unified machine that tier competes for the GPU's DRAM (C.4), so it matters only past the memory, for
the NVMe-backed tier. It is not on the path to a fast decode, so a NEON engine is not proposed ahead of the Metal one.

### B.1 `qdot` off x86, in short

- Every AVX2 item is under `cfg(target_arch = "x86_64")` (`crates/qdot/src/lib.rs:62`, 56 `#[target_feature]` fns in `lib.rs`, 5 in `generic.rs`).
  Off x86, `supports` is "the type has a kernel" (`lib.rs:224-228`), `has_features` is false (`:277-280`), `quantize_col` and `dot_row` run the scalar
  mirrors (`:431-434`, `:595-602`, `:605-636`), and the tile, row-lane and pack kernels are `unreachable!("no qdot … off x86_64")`
  (`:893-902`, `:1133-1136`, `:1269-1272`, `:1377-1380`), guarded by `has_tile`/`has_lanes`, which are false (`:747-775`, `:1159-1163`). So every type
  takes the fused path through its scalar mirror: the AVX2 kernels' bits, not their speed.
- Two kinds of arm. The **hand kernels** are the 16 types `dot_row_scalar_ty` matches (`lib.rs:610-627`: Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q8_0,
  Q2_K, Q3_K, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ3_S, MXFP4, IQ4_NL, IQ4_XS, Q2_0), each with an AVX2 body and an `emul`/scalar mirror, plus the tiles
  (14 types, `tile_kind` :747), the Q3_K row-lane tile (:1048), the Q4_K/Q5_K/Q6_K lanes (:1161-1292) and the AVX-512 bodies for Q4_K (6 `target_feature`
  fns, `q4k_avx512` :292). Their non-x86 arm is the mirror or a refusal, as above. The **runtime-detected helpers** take an AVX2 path when
  `is_x86_feature_detected!` says so and fall through to a scalar loop otherwise, so off x86 they are the scalar loop with no `cfg` arm of
  their own: `card_q8_1_fill` (:2596), `card_q3k_dot_row_cols` (:3068), `card_q4k_dot_row_cols` (:3268), `card_swiglu_clamp` (:3493), `card_expf` (:3521),
  `q_nope2_cells` (:9245, with a `cfg(not(x86_64))` arm at :9273), `dot_f32` (:9389), `sum_sq_f64` (:9452), `swiglu` (:9499), `swiglu_clamp` (:9524).
- The generic kernels (`crates/qdot/src/generic.rs`: `dot_avx2`/`dot_scalar`, Q8_K and Q8_2_X4 forms) serve no type today:
  `ENTRIES` is empty (`generic.rs:129`), the three decoders (`Q2k`, `Iq4Xs`, `Iq4Nl`, `:572`, `:606`, `:636`) are the gate's proof types, and
  `crates/qdot/tests/generic.rs:25` is `#![cfg(target_arch = "x86_64")]`, so the gate does not compile on a Mac.
- **NEON form of the two generic kernels (prototype, measured).** The AVX2 steps map as: `maddubs(|w|, sign(q,w))` then
  `madd(1, ·)` to `vdotq_s32` (signed × signed, the sign fold is not needed); `madd(scale, pair)` to `vmlaq_n_s32`
  of the per-16-value dot lanes; `_mm256_fmadd_ps` to `vfmaq_f32`; `_mm256_cvtepi32_ps` to `vcvtq_f32_s32`; `hsum_float_8` to
  `vaddq_f32(lo, hi)`, `vadd_f32(low, high)`, then lane 0 + lane 1; the Q8_2_X4 group fold to a `vpaddq_s32` chain. The integer
  sums are exact, and `maddubs` saturates only above 32767, which `|w| ≤ 127` and `|q| ≤ 127` (2 × 127 × 127 = 32258) never reach, so
  `vdotq_s32` equals `maddubs` + `madd` on the types' legal range. The float closes are the same fused multiply-adds in the same lane order, so the result
  is bit-identical. A scratch prototype (`target/mac-portmap/neonproto/main.rs`, built in the layer-3 copy against the real
  `generic::Entry::dot_scalar`) gave **0 mismatches on 2400 rows** (Q2_K with mins, IQ4_XS, IQ4_NL; random, spiked and
  alternating-sign activations; `target/mac-portmap/neonproto-run.log`). FAIL-first: an unfused multiply then add in the closing step
  gave 341 mismatches, a sequential lane sum for `hsum` gave 1292 (`neonproto-mut1-run.log`, `neonproto-mut2-run.log`). The
  intrinsics are Arm's (https://arm-software.github.io/acle/neon_intrinsics/advsimd.html : "int32x4_t vdotq_s32 ( int32x4_t r, int8x16_t a, int8x16_t b) r -> Vd.4S a -> Vn.16B b -> Vm.16B SDOT Vd.4S,Vn.16B,Vm.16B") and compile with no feature gate on the
  pinned nightly (`#[target_feature(enable = "dotprod")]`). Apple lists the capability as https://developer.apple.com/tutorials/data/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics.json :
  "hw.optional.arm.FEAT_DotProd": "Advanced SIMD Int8 dot product instructions."; `FEAT_I8MM` ("Advanced SIMD Int8 matrix multiplication
  instructions.", the `vmmlaq_s32` family) is 0 on this M1 Pro (measured) and not tested.
- What is missing for a CPU engine at speed: the type decoders for the hot types (only 3 exist, `ENTRIES` empty), a NEON tile kernel (up to
  `TILE_COLS` columns, so a prompt batch reads each weight block once), and the hand kernels' lanes for Q4_K/Q5_K/Q6_K. The Spark's decision
  stands (`dgx-spark.md` section 11, answer 1): "NEON only when a host-tier gate's wall asks for it." No Metal step waits for it.

### B.2 `serve::Engine` and what the Metal engine needs from it

`trait Engine` is `crates/serve/src/engine.rs:207-538`: 6 required methods and 32 with a default. The only implementation over a
real model is `SeatEngine` (`crates/gpu-gates/src/bind.rs:1519`, impl :2983), which drives a `Seat` (:1268, 12 required methods); the mock
is `crates/serve/src/mock.rs`. `crates/model` has no implementation, and `crates/serve/src/bin/bloomery-serve.rs:1` serves the mock only.

| Required method (`engine.rs`) | What a CPU or Metal engine needs | What exists today | Missing |
|---|---|---|---|
| `tokenizer` :209 | an `Arc<dyn serve::Tokenizer>` over the file's vocabulary | `tokenizer::Tokenizer::from_gguf` (`crates/tokenizer/src/lib.rs:112`, pure); the adapter `Vocab` is `crates/gpu-gates/src/bind.rs:81-173`, in the device-linked crate | lift `Vocab` and `VocabDecoder` (:174) into a device-free crate: about 140 lines `[derived: bind.rs:81-220]`, a move |
| `prefill` :219 | evaluate ids, no token | CPU: `deepseek2::forward::step` (`crates/model/src/arch/deepseek2/forward.rs:307`) over a `KvCache` (`kv.rs:76`); GPU: `Seat::prefill` (`bind.rs:1276`) | CPU: every architecture but deepseek2. Metal: the engine itself |
| `next` :234 | evaluate `last`, return the argmax, optionally the logits | CPU: `step` then `argmax` (`forward.rs:267`); GPU: `Seat::step` :1296, `logits_into` :1298 | as above |
| `reset` :383 | drop the cache | `Seat::reset` :1421; CPU: a new `KvCache` | none |
| `ctx_max` :409 | cache positions | `Seat::ctx_max` :1272 | none |
| `describe` :411 | device and position for a crash report | `SeatEngine`'s impl | a Metal device string |

The defaults carry a one-slot, no-draft, no-cache-reuse server (`slots` 1, `advance` = one `next`), so a Metal engine can answer
`bloomery-serve` with the six methods and `Seat`'s required twelve, of which `snapshot`/`resume`/`rollback`/`note`/`splits`/`keep`
can start as named refusals. Which architectures run end to end on the CPU path today: **only `deepseek2`** (`crates/model/src/bin/bloomery-decode.rs:33`
`RUNS = ["deepseek2"]`; `forward.rs` has `forward`, `step`, `new_cache`). The other architectures in `crates/model/src/arch/` hold
descriptions (`spec`, `place`, `roles`, `names`) and the host tier's expert legs (`host.rs` of `deepseek41`, `qwen35moe`, `glm5next`, `mimo2`);
`qwen3moe` has 869 lines and no forward. There is no host GQA attention, rope for GQA or QK-norm in `crates/model` (the only rope is the
MLA one, `deepseek2/attn.rs:326`), so a CPU engine for Qwen3-MoE is new code, not a wiring.

**The threads pool on macOS.** The pool pins workers across L3 groups read from sysfs (`crates/threads/src/lib.rs:12-24`, `topology.rs:78-140`).
On a Mac: no sysfs, so one group of one cpu and a default of one thread (`topology.rs:99-112` `[derived]`); no affinity (`KERN_NOT_SUPPORTED`,
measured); two core classes (`hw.nperflevels` 2: 8 performance and 2 efficiency cores here, measured) and chunks split evenly
(`lib.rs:735-742`), so a worker on an efficiency core stretches every dispatch. The Mac form: the thread count is
`hw.perflevel0.logicalcpu`, workers take `QOS_CLASS_USER_INTERACTIVE`, topology is one group of the performance cores (their shared L2,
12 MiB here, stands where the L3 stands) `[derived: a design, not measured]`.

## Part D. Gates on a Mac (M3)

**Status (main `d916e20ac`).** The Metal gates run on a Mac through `tools/mac-gate.sh`: `just lab-metal` runs the
device gate, the `qwen35moe` kernels gate and that gate's mutants, and `just lab-metal-e2e` runs the `qwen35moe` session
against ik's reference sets of the fixture family. They are `lab-` recipes, so no landing batch selects them. No Mac
timing runner exists. The tables and bullets below are the plan of 2026-10-21.

**What can run at each phase.**

| Phase | Runs on a Mac | Evidence class |
|---|---|---|
| now | `just mac-check`, `mac-lint`, `mac-fmt-check`, `mac-static`, `mac-test` (`tools/mac-check.sh test`: 14 pure crates, 1477 passed, 0 failed, 210 ignored, 52 test binaries, measured, `target/mac-portmap/mac-check-test.log`) | development-loop evidence, never landing evidence (`AGENTS.md`) |
| M0 | the host library tests of the 7 crates that fail today (`threads`, `qdot`, `engram`, `model`, `model-hf`, `gguf`, `engram-lab`), natively; `qdot`'s non-hardware tests (the `hw_` ones are `#[ignore]`d: 62 of 94 in `crates/qdot/tests/qdot.rs`) | the pure-crates rule's native test, extended to those crates |
| M1 | `qdot` bit tests of a NEON kernel against its scalar mirror (the prototype's clause, above); the CPU engine's gates | needs `tests/generic.rs` un-gated from x86 (:25) or a new file |
| M2 | one Metal family gate: the first model's layer taps and token ids against the box's CUDA reference (a reference set dumped once on the box and read by `crates/refset`, which already runs on the Mac); each kernel against its scalar or CPU oracle (`qdot`'s q8_1 and Q4_K mirrors for the integer-exact ones) | a gate that needs Apple Silicon and a Metal device |
| M3 | fixture-tier Metal gates in a batch; a timing runner | a Mac runner (below) |

**What assumes Linux, and where.**

- `tools/gate.sh:74,97` runs `timeout --kill-after=10`; `tools/gpu-gate.sh` takes `flock` locks on `/root/...`, reads `/proc/<pid>/stat`
  and `nvidia-smi` (its header :16-17 says the self-test needs them and the Mac has neither; the self-test already carries Python stand-ins, :381-400).
  `tools/box.sh` is the box. None of these run a gate on a Mac; a Mac gate runner is new: `tools/mac-gate.sh`, the lock a `lockf`-style
  stand-in, the bound `gtimeout` or the Python one.
- The timing runners (`tools/ref/*`, `lease_take`) are the box's: a machine-wide lock, GPU-idle wait and witness blocks. A Mac equivalent
  needs its own witness (chip, memory, power mode, thermal state, other GPU users): which Apple tool gives a usable thermal and power
  reading without privilege was not tested `[unverified]`. Until it exists, `AGENTS.md` "Never time by hand" forbids a Mac number, so
  the two terms of C.5 wait for it.
- `tools/recipes.py`: `pure-crates` (:5876) and its `_NOT_ON_MAC` list (:5512-5525) — its `#[target_feature(` pattern must take
  a feature name (`neon`, `dotprod`, `i8mm`) as fine on aarch64, and the name list needs `O_DIRECT`; `HOST_DEVICE_LINKED` (:5905) holds one
  crate (`bloomery-app`) and concerns the Linux host nightly (`tools/nightly/run.sh`), not the Mac; the `gate-*` recipes all start `./tools/box.sh`
  (`justfile:1609`, `:1655`), so a Mac recipe needs its own `mac-` prefix, or a `BLOOMERY_LOCAL` mode as `dgx-spark.md` section 4 has for local gates.
- Crate classes: a new crate (`bloomery-device`, the platform crate, a Metal crate) needs rows in `tools/gate-class.tsv`, `tools/families.tsv`,
  `tools/owners.tsv` and `just check-recipes` passing `[unverified: not run]`.

**The `AGENTS.md` "Never" bullet that would change** (proposed text; `AGENTS.md` is not edited here). Today:

> **Never run a gate where it cannot run.** Gates need x86_64 Linux and an NVIDIA card: `qdot` uses `std::arch::x86_64`, `threads` Linux affinity calls. On an arm64 Mac run only the static tier: …

Proposed, from M0:

> **Never run a gate where it cannot run.** The CUDA gates need Linux and an NVIDIA card. The host gates (`just mac-test`, the crates `tools/recipes.py pure-crates` selects) run on Linux and on aarch64 macOS; every Linux-only call sits behind `bloomery-os`, and a Mac result is development-loop evidence. From M2, the Metal gates (`gate-metal-*`) need Apple Silicon and run through `tools/mac-gate.sh`; they never run on Linux. On an arm64 Mac without the Metal tier run only the static tier: …

And the first bullet ("Never build a device crate with plain cargo") gains one sentence at M2: "A Metal crate links no cuda-core and builds with plain cargo; `tools/recipes.py` treats it as a pure crate only when it builds without a Metal device."

## Part E. The phased plan

**Status (main `d916e20ac`).** M0a and M0b landed as `crates/os` (`cd40ceac9`, `000a5d52d`, 2026-10-21). M2.0 was
measured (section C.5). The device contract and its Metal backend landed as `bloomery-device` and `bloomery-metal`
(`c16dd696c`, 2026-10-22). The first Metal family is `qwen35moe` (question 2 of Part F); its tensor-op GEMMs
(`aaf14ad24`, 2026-10-25) and the Apple Silicon release build (`e3ee4b796`, 2026-10-25) followed. The table below is the
plan as written and its rows are not edited.

Ordered for the shortest path to a fast Metal decode on one model. M1 leaves the critical path: the Metal engine runs the model whole on
the GPU out of unified memory, where a CPU tier adds no capacity (C.4), so nothing in M2 waits on NEON or a CPU `serve::Engine`; the
`Vocab` lift (B.2) is the only part of M1 the serve step needs. M0 stays: the Metal crate reuses `bloomery-model`'s placement and
hparams, `gguf`, `refset` and `threads`, which did not build on a Mac before `crates/os`.

| Step | Work items | Files touched; size `[derived]` | What proves it (change class, `AGENTS.md` "Derive first") | Depends on |
|---|---|---|---|---|
| **M0a** compile fence | the 45 errors at 38 lines: a leaf crate `bloomery-os` (name open, Part F) with `affinity`, `rusage`, `file_hint` (drop, populate, sync), `direct_open`, `rename_noreplace`, `mincore`, `free_bytes`; callers switched | 10 files, ~45 call lines changed; the crate ~15 functions × ~20 Linux-body lines + ~25 macOS lines ≈ 650 lines `[derived: counts × typical body]` | **move**: `just canon-check BASE` (both sides' `--no-jit` scans of the scan set equal; no kernel moved); the owning gates of the touched crates through `--round-ledger` (`gate-threads`, `gate-model-hf`, the engram and model gates); on the Mac `tools/mac-check.sh test` extended to the 7 crates | none |
| **M0b** runtime on macOS | `ROW_PAGE` a parameter (`placement/workstation.rs:141`, `model/.../workstation.rs:24-28`, `placement.rs:3055-3062`); `host_available` from `hw.memsize` and `vm_statistics64` or the Metal working set; the pool's default threads and QoS; `ru_maxrss` units | 6 files, ~120 lines | move on x86 (`canon-check`); a unit test at page 16384 and a Mac-only clause by name | M0a |
| **M2.0** two measurements | `c_d` (an empty-kernel ICB of 12 L + 4 dispatches replayed) and `f_bw` (a read-only kernel over 2 GiB), Swift or Rust probe, taken by a Mac runner | 1 probe crate, ~200 lines; the runner part of M3 | each with its prediction card named in the card's format, the expected value written from the datasheet bytes and the gap budget (C.4) | M3's runner, minimal |
| **M2.1** device layer | `bloomery-device`: `Buffer`, `Pinned`, `Stream`, `Event`, `Graph` (a CUDA graph; one command buffer a token on Metal), `HostFlags`, fault word, census; CUDA behind it, then Metal; family imports renamed | new crate ~1.5k lines (`crates/gpu/src/graph.rs` 1595 lines is the model); ~280 import lines in 9 crates | **move** on x86: `just canon-check BASE` plus the structural lines (graph node count, eager = replay, e2e set identical) | M0 |
| **M2.2** the decode set | the 15 kernels of C.2 in MSL, one at a time, each with a gate against its oracle: integer-exact kernels bit-equal, transcendental ones in a band | ~15 `.metal` files, 1.4k entry lines plus cores `[derived]`; one gate bin | **add**: the family gate green with FAIL-first per clause; the CUDA side unchanged (`just ptx-compare` rc 0) | M2.1 |
| **M2.3** the chain | `ChainOps` over `dispatch.rs`; the Metal chain in one command buffer a token, plain encoder, concurrent dispatch where kernels are independent; a `generate`-style CLI; layer taps against the box's CUDA reference | `dispatch.rs` split (1513 lines) into walk and ops; ~600 lines new | **move** for the CUDA side (`canon-check`, the structural lines), **add** for Metal (the e2e token set equal to the box's) | M2.2 |
| **M2.4** prefill | MPP `matmul2d` GEMM and grouped expert GEMM, prefill flash | the prompt-side set of C.2's last paragraph | the family gate; the timing on an M5-class machine | M2.3, an M5 machine |
| **M3** Mac tier | `tools/mac-gate.sh`, a Mac timing runner with witnesses, `AGENTS.md` text (Part D), a Mac table in the README with the chip named | 3 scripts, 1 contract edit | the runner's self-test (`tools/gpu-gate.sh`'s model); a number is admissible only from the runner | M2.0 for the minimal part |
| **M1** (off the critical path) | `Vocab` lift (~140 lines `[derived]`, with the serve step); NEON generic kernels + decoders for hot types; a CPU engine for one architecture | ~400 lines for NEON and the tests; the engine is new code (B.2) | **add** and **integer-path reorder** classes: bit-equal to the scalar mirror (prototype above) | M0 |

**The first implementation round (M0a), ready to use.**

- *Boundary.* One owner for the new crate, `Cargo.toml` `members` (:3), the callers below; no other agent edits them.
- *Files.* New: `crates/os/Cargo.toml`, `crates/os/src/{lib,affinity,file,mem}.rs`. Changed: `crates/threads/src/lib.rs` (:750, :809, :839),
  `crates/threads/src/helper.rs` (:216, :232), `crates/threads/tests/helper.rs`, `crates/gguf/src/lib.rs` (:1318), `crates/engram/src/{lib,direct}.rs`
  (:542, :701, :749; :35), `crates/engram/tests/engram.rs` (:871), `crates/model/src/{fileio,r8file}.rs` (:19, :33; :510, :525),
  `crates/model/src/placement/host_lock.rs` (:818, :901, :1207), `Cargo.toml`, `Cargo.lock`.
- *Rules.* Each Linux body moves unchanged into `crates/os` under `cfg(target_os = "linux")`; the macOS bodies are the forms of A.2; a function with no
  macOS form returns a named `Unsupported` error, never a defined value (`AGENTS.md`: no silent failure). No behaviour change on Linux.
- *Proof.* Before: record the base scans, `just narrow BASE` then `just canon-check BASE` (rc 0 = every entry the same program), on the box.
  After: `tools/mac-check.sh test` with the 7 crates added to the native set (expected: `threads`, `qdot`, `engram-lab`, `model`, `model-hf` build and their non-ignored tests run; the clauses skipped by name are
  listed), `just mac-static BASE`, then the owning gates through `tools/gate-batch.sh --round-ledger`.
- *Completion commands and expected output.*
  `cargo check -p bloomery-threads -p bloomery-qdot -p bloomery-engram -p bloomery-engram-lab -p bloomery-model -p bloomery-model-hf --all-targets` on the Mac: rc 0;
  `cargo test -p bloomery-gguf --no-run`: rc 0; `just canon-check <base>`: rc 0; `python3 tools/recipes.py pure-crates`: the 7 crates move from `rejected` to `pure` only after
  the `_NOT_ON_MAC` edit; `python3 tools/recipes.py --self-test`: green.
- *Report.* The per-crate rc table again, the list of clauses skipped on macOS by name, the `canon-check` line.

## Part F. Open questions for the maintainer

**Status (main `d916e20ac`).** The code answers three of these, not a maintainer's reply on this page. Question 2: the
first Metal family is `qwen35moe` (`crates/metal/src/arch/qwen3moe/arena.rs:1` is its arena), and Qwen3-30B-A3B is
refused by name. Question 7: macOS 26.2, refused by name below it (`crates/metal/src/device.rs:19`). Question 8:
`objc2-metal`, pinned by `=` (`Cargo.toml:39`), with no Objective-C++ shim. Question 5 has a partial answer: a
contributor ran the tensor path on an M5 Ultra ([`../mac.md`](../mac.md)); who owns an M5-class machine is not answered.
The other questions are not answered on this page.

1. **Numeric contract of the first Metal kernels.** q8_1 activations as on CUDA (integer-exact, bit-equal where the order is copied, two extra dispatches a layer for the quantizers) or f32/f16 activations as ggml-metal (fewer dispatches, no activation rounding, but a different family of bits and a band against ik that is not measured)? The kernel list depends on it.
2. **The first Metal family.** `qwen3moe` (15 kernels, no fixture row: `registry.rs:103`), or `qwen35moe` (fixture exists, `registry.rs:124`, plus the delta rule and gated kernels), or both with `qwen3moe` first and the fixture family as the gate input?
3. **The seam's shape.** The `ChainOps` cut over `dispatch.rs` (C.1), or a Metal copy of the chain until a second family arrives?
4. **Where the Linux-only calls live.** A new leaf crate `bloomery-os`, or a module in `bloomery-gguf` (the lowest crate they all reach)? (`threads` depends on `levers` only; `gguf` is below `engram` and `model` but not below `threads`.)
5. **An M5-class machine.** Tuning the tensor-op kernels and the two C.5 terms need one (the M1 Pro cannot show the Neural Accelerators' rate; ggml's own comment says the tensor API does not pay before M5). Who owns it, and how is it reached, as the box is?
6. **A Mac runner and witness.** `tools/mac-gate.sh` and a timing runner with a witness block: what reading of power and thermal state is accepted as the witness (`AGENTS.md` "Never time by hand" has no Mac form)?
7. **Minimum macOS.** MLX says the Neural Accelerator path needs "macOS 26.2 or later" (the research page above); Metal 4 tensor ops and `MTLResidencySet` set a floor of their own. Pin one?
8. **Metal bindings.** `objc2-metal` (https://github.com/madsmtm/objc2 , pushed 2026-10-17; its `Cargo.toml` lists `MTL4ComputeCommandEncoder` and `MTLTensor` features, the published crate's coverage `[unverified]`) or a small Objective-C++ shim as `ggml-metal` has? (https://github.com/gfx-rs/metal-rs says "Use of this crate is deprecated".)
9. **Is M1 dropped from the critical path** as proposed, with the `Vocab` lift the only part the serve step needs, and NEON only if a host-tier gate's wall asks (`dgx-spark.md` section 11, answer 1)?
10. **Public numbers.** One table a chip (`AGENTS.md`: "a table never mixes cards"), and the reference engines in the same session. Which Mac engines are the line to beat is a separate round's survey; the baseline rule for the README (llama.cpp, mistral.rs) stands.
11. **Adapting ggml-metal source.** MIT; a kernel taken from it carries the notice in `THIRD_PARTY_NOTICES.md`. Acceptable for the K-quant mat-vec structure?
12. **Release.** An aarch64-apple-darwin archive needs signing and notarization; who holds the keys (the secrets go to the vault, not a file)?

## Appendix. What was run in this round

All under the round's `target/mac-portmap/` (gitignored, not published):

| What | File |
|---|---|
| native census, layer 1 | `check/*.log`, `check/*.rc` (21 crates) |
| layers 2 and 3 (scratch copies, stubbed libc) | `check-layer2/`, `check-layer3/` |
| the error sites, deduplicated | `sites.tsv` |
| `/proc` and `/sys` lines | `proc-sys-sites.txt` |
| kernel index (`#[kernel]`, 376 at line start; 385 grep lines count 9 comment mentions) | `kernels.tsv` |
| native tests of the pure crates | `mac-check-test.log` |
| NEON prototype, its run, two mutants | `neonproto/main.rs`, `neonproto-run.log`, `neonproto-mut1-run.log`, `neonproto-mut2-run.log` |
| MSL compile probe on the M1 Pro | `msl-probe.log` |
| MPP `matmul2d` probes | `mpp-int8-probe.log`, `mpp-types-probe.log`, `MPPTensorOpsMatMul2d.h.txt` |
| machine and device probes (sysctl, `MTLDevice`, thread affinity, QoS) | `machine-probe.log`, `probes/{dev.swift,aff.c}` |
| MSL and MPP probe sources | `probes/{msl,mpp,mpp2}.swift` |
| pages fetched for the sourced lines | `web/` |
| `man` lines used | `man-facts.txt` |
