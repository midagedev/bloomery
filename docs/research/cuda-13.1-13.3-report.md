> 원문 조사 보고(영문) — opus 라운드 `cudares`, 2026-09-23 새벽. 사용자 질문 "업데이트하면서 추가된 명령어 세트나 기능이 있는지"에 대한 답이다. 측정은 없고 헤더·릴리스 노트·PTX ISA 문서·박스의 파일을 읽은 것이다. 그 뒤 리드가 속성 122·123·148·152를 두 카드에서 읽었다(측정 — rig-log [09-23#cuda-13x-features](https://github.com/midagedev/rig-log/blob/main/log/2026-09-23.md#cuda-13x-features)). 계획에 남은 것은 `plan.md`의 B2 카드(합류 후보 넷)와 「GPU 선 트리아지」의 계기 항목이다.

# CUDA 13.1–13.3 vs 13.0 for bloomery on sm_86 (round `cudares`)

## 1. Answer in five lines

1. **A toolkit switch cannot change the SASS we run through the PTX path.** Every module in `generate`, `gate_p8` and `gate_e2e` is PTX `.version 7.1 .target sm_86`, written by LLVM 21.1.8, whose NVPTX backend stops at `ptx88`. The R615 driver JIT turns that PTX into SASS, so the driver moves our SASS, not the toolkit. The toolkit's only input is libdevice. At least 19 of its 352 functions changed, none that we call, which is why the `.oxart` is byte-identical. One exception: an NVVM-IR or LTOIR payload would load the toolkit's libNVVM or nvJitLink. `generate` carries the loader code for both, but no module uses it.
2. **PTX 9.1–9.4 adds nothing we can reach on sm_86.** The sm_86-legal additions are `clmad`, the `mma_throughput` pragma, `.mmio` acquire/release and `ld/st.volatile.local`. All of them need `.version` ≥ 9.1, which LLVM 21 cannot emit. The packed-byte integer SIMD in 9.2 (`.s8x4`/`.u8x4`) is sm_107f/sm_120f only.
3. **The one real lever is for B2: host-task spin-wait** (13.2: `cuLaunchHostFunc_v2`, and `syncMode = CU_HOST_TASK_SPINWAIT` on host nodes). The R615 `libcuda` already exports it. Our bindings need 13.2+ headers to call it. A round should measure it against blocking host nodes and against `cuStreamWaitValue32`, and must count the core that the spinning driver thread takes from the pinned CPU pool. A second candidate is the memop atomic reduction (13.1), gated by device attribute 148, which has not been queried yet.
4. **Nothing lowers the decode graph's per-node cost on sm_86.** PDL is sm_90+. Recapture (13.3), graph and node IDs (13.1) and `cuGraphNodeGetParams` (13.2) are conveniences and instrument aids.
   - For B3, GDS P2PDMA is plausible only on the A6000, because GDS mode is limited to Data Center and Quadro cards. It is unverified.
   - GDS also works in 4 KiB units and bypasses the page cache, which works against B3e.
   - The 13.3 DMA-BUF `mmap()` of VRAM (write-combined on x86, attribute 152) is a new CPU→VRAM path worth measuring.
5. **The instruments moved tonight.**
   - `/usr/local/cuda/bin/ncu` now runs Nsight Compute 2026.2.1. In that version, "Local Memory Spilling Requests", a column the runner keeps, no longer counts shared-memory spills.
   - nsys stays at 2025.3.2, which has no host-function trace; B2 will need it.
   - ptxas 13.3 moves 4 of 52 kernels by ±1 register, so ptx-scan tables depend on `PTX_SCAN_PTXAS`.
   - cutile-rs can now target sm_86: Tile IR 13.2 bytecode with 13.3 `tileiras`, which adds i4 storage and gather/scatter. The i8 `mmai` op binds since V13_2. A code search found no dp4a.

## 2. Findings table

| # | Feature | First toolkit | sm_86? | Reaches us via | Plan row | What a round would measure | Source | Tag |
|---|---|---|---|---|---|---|---|---|
| F1 | **Host-task spin-wait**: `cuLaunchHostFunc_v2(stream, fn, userData, syncMode)`; `CUDA_HOST_NODE_PARAMS_v2.syncMode`; `CU_HOST_TASK_SPINWAIT = 1` ("The execution thread will spin wait until new host tasks are ready to run"). `cuGraphAddHostNode` still takes the v1 struct, which has no `syncMode`. The v2 struct is reachable through `cuGraphAddNode` (`CUgraphNodeParams.host`). The host fn "must not make any CUDA API calls". The call can return `CUDA_ERROR_NOT_SUPPORTED`. | 13.2 | No arch restriction in the header or the release notes; not probed | Driver: `libcuda.so.615.71.09` exports it. Bindings must be regenerated from 13.2+ `cuda.h`. | B2 | GPU idle gap per join (producer kernel end → consumer kernel start) for three mechanisms: blocking host node, spin host node, `cuStreamWaitValue32` on a host-written flag. Also CPU-tier tok/s with the spinning driver thread present vs absent. | cuda133.h:435-438, 1964-1968, 4971, 19325, 19383, 20426; 13.2 RN §General: "Host tasks now support a new spin-wait dispatch mode, which can reduce execution latency, in addition to the existing mode that blocks on an interrupt from the GPU." | [read]; core contention [inferred] |
| F2 | **Stream-memop atomic reduction**: `CU_STREAM_MEM_OP_ATOMIC_REDUCTION = 8` in `cuStreamBatchMemOp`; ops OR/AND/ADD on u32/u64 | 13.1 | Gated by `CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED` (148); not queried | Driver (`cuStreamBatchMemOp_v2` exported) + 13.1+ headers | B2 (counter join: k producers ADD, one WaitValue GEQ k) | Attr 148 on both cards. If it is 1: join latency vs k separate WriteValues. | cuda133.h:577, 593-605, 640-648, 967; 13.1 driver-API TYPES page | [read]; sm_86 support unknown |
| F3 | **Existing v2 memops** (B2 context, not new): 32-bit `cuStreamWaitValue32_v2`/`WriteValue32_v2` have no capability attribute. 64-bit ops = attr 122; `WAIT_VALUE_NOR` = attr 123. v1 attrs 92–94 are deprecated. The box's `/proc/driver/nvidia/params` shows `EnableStreamMemOPs: 0`. | ≤13.0 | Only a runtime call proves support | Driver (exported) | B2 | One `cuStreamWaitValue32` + `WriteValue32` call on each card; attrs 122/123 | cuda133.h:910-912, 941-942, 18176-18186; box params | [read]; that `EnableStreamMemOPs` governs only the v1 path is [inferred] |
| F4 | **Recapture into an existing graph**: `cuStreamBeginRecaptureToGraph`. RN: "The nodes must be recaptured in the same order as the original source graph… fail if the recaptured topology diverges". Error `GRAPH_RECAPTURE_FAILURE` = 918. | 13.3 | No arch restriction | Driver (exported) + 13.3 headers | Decode graph; A4 (flash grids sized by `segments_for(ctx_max)`) | At a ctx change, compare three paths: recapture + `cuGraphExecUpdate`, `cuGraphExecKernelNodeSetParams` on the 54 flash nodes, and re-instantiate. The payoff is bounded by A4's dead-segment time. | cuda133.h:3356, 16629-16631, 16691; 13.3 RN §General | [read]; "convenience, not a new capability" [inferred] |
| F5 | **Graph/node identity**: `cuGraphGetId`, `cuGraphExecGetId`, `cuGraphNodeGetLocalId`, `cuGraphNodeGetToolsId`, `cuGraphNodeGetContainingGraph` | 13.1 | No restriction | Driver (exported) + headers | Instrument (`nsys-gpu.sh`) | Whether `cuGraphNodeGetToolsId` equals CUPTI `graphNodeId` for our 648 nodes; if so, nsys rows key to nodes directly | cuda133.h:21502-21589; 13.1 GRAPH page (WebFetch) | [read]; ToolsId == CUPTI id [inferred] |
| F6 | **`cuGraphNodeGetParams`** (generic parameter read-back) | 13.2 | — | Driver (exported) + headers | Decode-graph structural lines | Assert per-node grid and params after capture, as a structural ratchet; no timing needed | cuda133.h:23063; 13.1 vs 13.2 GRAPH pages (WebFetch) | [read] |
| F7 | **In-graph control flow**: IF/WHILE/SWITCH conditional nodes and device-updatable kernel nodes are **already in 13.0**. 13.3 adds `CU_FUNC_ATTRIBUTE_DEVICE_NODE_UPDATE_SUPPORTED` (16). The runtime's `cudaGraphConditionalHandleCreate_v2` adds a ctx parameter. | 13.0 / 13.3 | — | — | Decode graph | Nothing new from this window | cuda130.h:1862-1918; cuda133.h:1176; cuda_runtime_api.h (13.3):13385 | [read] |
| F8 | **Launch attr `SHARED_MEMORY_MODE` (18)**: "Valid for graph nodes, launches". `ALLOW_NON_PORTABLE` permits up to `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`. Also new: `PORTABLE_CLUSTER_SIZE_MODE` (17). | 13.2 | Shared-memory mode yes [inferred]; clusters are sm_90+ [inferred] | Driver + headers | A4e (56 KB dynamic smem) | Nothing to measure: a per-launch alternative to `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` | cuda133.h:2194-2208, 2378-2381 | [read]; applicability [inferred] |
| F9 | **Copy-engine steering**: `CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` is now honoured on non-Tegra for the batched memcpy APIs ("the driver favors Copy Engines to overlap transfers with compute and keep SMs free"). Plus the single-copy `cuMemcpyWithAttributesAsync` / `3D`. | 13.1 / 13.2 | No restriction stated | Driver (exported) + headers | A6 residue (`copy_from_host`); B3 row upload | H2D of the per-token rows with and without the flag (48 × 272 B ≈ 12.75 KiB, derived): overlap with decode kernels in nsys, and step Δ | 13.1 RN §General (raw); cuda133.h:11504, 11529 | [read] |
| F10 | **DMA-BUF `mmap()` of device memory**: attr `DMA_BUF_MMAP_SUPPORTED` (152). "For device memory on x86 systems the mapping will be a write combined mapping." `DMA_BUF_MAPPING_TYPE_PCIE` maps via BAR1. | 13.3 (R610+) | RN says "discrete GPU video memory"; attr not queried | Driver (R615) + 13.3 headers | B3 (CPU stores rows straight into VRAM) | Attr 152. CPU store bandwidth into a WC mapping for 48 scattered 272 B rows per token. BAR1 is 256 MiB on the 3090 (`EnableResizableBar: 0`) and 64 GiB on the A6000. | cuda133.h:969, 12868-12905; 13.3 RN §General; box `nvidia-smi -q` | [read]; fit for B3 [inferred] |
| F11 | **GDS / cuFile 1.18.1.6**. P2PDMA mode since GDS 1.13: no `nvidia-fs.ko` needed for ext4/XFS on NVMe; kernel ≥ 6.2. Open kernel module only. GDS mode only on "Data Center and Quadro (desktop) cards with compute capability > 6"; all other cards get compatibility mode. Unaligned I/O (offset, size or buffer not 4K-aligned) goes through bounce buffers. Small BAR: "do not register the buffer". | P2PDMA from GDS 1.13; 13.3 ships 1.18.1.6 | A6000 GDS-eligible [inferred]; 3090 compat only [inferred] | Library + kernel. Box: kernel 6.8.0-139, `CONFIG_PCI_P2PDMA=y`, `nvidia-open` 615.71.09, no `nvidia-fs`, `/models` ext4 on `nvme0n1p1`. | B3, B3e | `gdscheck -p` first. Then per-token latency of about 51 aligned 4 KiB reads (B3c's 51.0 pages/token ≈ 204 KiB/token, derived) into A6000 memory, vs pread + H2D. GDS bypasses the page cache, which B3e's cache relies on. | GDS RN and best-practices guide (both raw); box `uname -r`, `/boot/config-*`, `df -T` | [read]; per-card eligibility [inferred] |
| F12 | **Green contexts**. 13.1: `cuDevSmResourceSplit` ("Splits a CU_DEV_RESOURCE_TYPE_SM resource into structured groups"), `cuStreamGetDevResource`, workqueue config and scope. 13.3: `CU_GREEN_CTX_NONE`; the default stream is now optional. | 13.1 / 13.3 | **Yes.** 13.3 header: "On Compute Architecture 7.X, 8.X…: The smCount must be a multiple of 2… alignment… is 2". Coscheduled max is 2 below 9.0. 13.0 said 8.X "minimum count is 4 SMs". | Driver (exported) + headers | No plan row | Decode tok/s @ n, depth, A6000 with the graph in an N-SM green context. The header gives no concurrency or forward-progress guarantee. Grid-barrier kernels must size grids to the context's SM count. | cuda133.h:26456-26458, 26470-26482, 26880-26912, 26981, 27250; cuda130.h:24714; 13.3 RN; 13.4 RN cuSOLVER (a hang from exactly that) | [read]; relevance [inferred] |
| F13 | **PTX ISA 9.1/9.2/9.3/9.4** (toolkits 13.1/13.2/13.3/13.4). sm_86-legal: `clmad` (9.3, sm_80+), `.pragma "mma_throughput"` (9.3), `.language` (9.3), `ld/st.volatile.local` (9.1), `ld.acquire`/`st.release` `.mmio` (9.3, sm_75+). Not sm_86: see §4. | 13.1–13.3 | As listed | Toolkit ptxas (AOT) or driver JIT (our PTX); but our emitter stops at PTX 8.8 | B4 | Nothing is reachable without an emitter that writes `.version` ≥ 9.1 | ptx_isa_9.3.pdf, ptx_isa_9.4.pdf; LLVM 21.1.8 `llc -mattr=help` | [read] |
| F14 | **libdevice 13.0 → 13.3**: ≥19 of 352 functions changed. CUDA Math 13.2 changed `expm1f`/`erff`; 13.3 fixed `__mul24` with constant inputs. | 13.2 / 13.3 | n/a | Toolkit, at build time only | None | None: the functions we call (`expf`, `sqrtf`, `roundf`, `exp2f`…) are unchanged | Box libdevice.10.bc from both toolkits; 13.2/13.3 RN §Math (raw) | [read] |
| F15 | **ptxas 13.3 register allocation**: 4 of 52 `generate` entries move by ±1 register (`expert_gate_up_swiglu_q3k` 38→39, `gate_up_swiglu_q3k` 38→39, `q3k_gemv_heads` 39→40, `q3k_gemv_heads_pair` 40→39). 0 spills in both versions. | 13.3 | Yes | Toolkit tool, used only by `just ptx-scan` | ptx-scan (AGENTS.md change-class proof) | Re-baseline once if `PTX_SCAN_PTXAS` moves; never compare tables made with different ptxas | Box `ptxas -v`, both toolkits | [read] |
| F16 | **Nsight Compute 2026.2.1 now runs behind `/usr/local/cuda/bin/ncu`**. `--clock-control` default is boost (2026.1). Boost locking on Ampere+ (2025.4). Node-level profiling of conditional and device-updatable graph nodes (2025.4). Green-context profiling fix (2026.2.1). Spill metrics redefined (§7). | 2025.4–2026.2 | Yes | Tool | `ncu-gpu.sh` | Tables that span tonight mix two instruments; the `[config]` line records which | ncu RN (raw); box section files | [read] |
| F17 | **Nsight Systems 2025.6+**: "CUDA Host Function Trace – Added trace support for CUDA Graph host function nodes and cudaLaunchHostFunc()". Hardware trace is the default "when supported". Export row order is no longer guaranteed. 2026.1 removed text and JSON export. | 2025.6 / 2026.1 | Host-function trace yes [inferred]; the HW-trace note names Blackwell [read], so sm_86 likely falls back to software trace [inferred] | Tool: 2026.1.3 is installed, but the 13.0 wrapper pins 2025.3.2 | `nsys-gpu.sh`, B2 | B2's host-node spans need 2026.1.3 called explicitly. Our sqlite export sorted `ORDER BY start` is unaffected. | nsys RN 2025.6 and 2026.1 (raw); box wrapper scripts | [read] |
| F18 | **CUPTI 13.3**: `CUPTI_API_VERSION` 130301 (13.0: 130001); `CUpti_ActivityKernel12` (13.0: 10); `CUPTI_FUNC_EXECUTION_MODEL_TILE`. `cuptiGetGraphNodeId` exists in both. | 13.3 | Yes | Tool library | Instrument | None | Box CUPTI headers | [read] |
| F19 | **13.3 compiler fix**: a reconvergence miscompile "present since CUDA 12.8" in kernels "that contain two or more nested levels of thread divergence where the compiler elided convergence instructions" | 13.3 | All archs [inferred] | Toolkit nvcc/ptxas (ik's ahead-of-time SASS); the driver JIT for us [inferred] | ik baseline | Open question: do any of our reference data come from ik's GPU outputs, or only its timings? | 13.3 RN §2.4.1 (raw); box `libnvidia-ptxjitcompiler.so.615.71.09`, dated Sep 5 | [read]; reach [inferred] |
| F20 | **cuBLAS 13.6.0.2** (13.3.1). Green contexts supported from 13.3. INT8→INT32 with N > 65,536 or batch > 1 fixed in 13.1. Two known issues introduced in 13.3 are still listed in 13.3.1: GEMV-like calls (M or N = 1) "may return CUBLAS_STATUS_NOT_SUPPORTED for certain shapes and workspace configurations", and `cublasLtMatmulAlgoGetHeuristic()` during capture of a non-default blocking stream "returns CUBLAS_STATUS_INTERNAL_ERROR". | 13.1–13.3.1 | No cuBLAS bullet from 13.0 through 13.3U1 names Ampere, sm_8x or compute capability 8 | Library (ik; RUNPATH pins 13.0 today) | ik baseline | See §5 | 13.3.1 RN §cuBLAS (raw) | [read] |
| F21 | **CUDA Tile / cutile-rs on sm_86** (see §6) | Bytecode 13.2; toolkit announcement 13.3 | Yes: i8–i64 supported, no f8 | Toolkit `tileiras` 13.3 (lists sm_86), or the driver JIT of bytecode | A5, B4 | An `mmai` i8 tile GEMM vs our IMMA `mma.sync` at A5 shapes. Tile shape decides whether it uses tensor cores or FFMA. | Tile IR 13.2 RN and 13.3 stability page (raw); cutile-rs compatibility (WebFetch); `_core.rs:2537-2559` | [read] |
| F22 | `cuFuncGetParamCount` / `cuKernelGetParamCount` | 13.2 | — | Driver (exported) + headers | None | None (it would let cuda-host check kernel arity at load) | cuda133.h:8747, 18704 | [read]; use [inferred] |
| F23 | `CU_JIT_BINARY_LOADER_THREAD_COUNT` (35); env `CUDA_BINARY_LOADER_THREAD_COUNT` | 13.1 | — | Driver | None | Cold module load only | cuda133.h:1629 | [read] |
| F24 | Coredump callbacks (4 functions) | 13.2 | — | Driver (exported) + headers | None | A debug aid for illegal-address faults; not for Xid 79 | cuda133.h:26348-26417 | [read]; usefulness [inferred] |

"Exported" means that `nm -D --defined-only libcuda.so.615.71.09` lists the symbol. I checked 20 new entry points and all are present [read].

## 3. Header diff summary (13.0 → 13.3)

Method: `box:/usr/local/cuda-13.{0,3}/include/`. Declarations are counted by regex. Line numbers are in the 13.3 files; copies of all six headers are in the scratchpad.

**`cuda.h`.**
- `CUDA_VERSION` 13000 → 13030.
- Functions 499 → 533: **+34, −0**.
- Deprecated functions 49 → 49.
- Versioned `#define` remaps: 47 → 47, the identical set. For example `cuStreamBeginCapture` → `_v2` is at :150 and `cuGraphInstantiate` → `WithFlags` at :159.
- About 70 new enum members and constants (the regex counted 71, ±2 artifacts).
- One internal member removed: `CU_DEV_RESOURCE_TYPE_MAX`.

| Area | New | First toolkit | Lines |
|---|---|---|---|
| Host tasks | 1 function (`cuLaunchHostFunc_v2`, PTSZ define at :219); enum `CUhostTaskSyncMode` (2 members); `syncMode` field | 13.2 | :435-438, :1964-1968, :19383 |
| Stream memops | `ATOMIC_REDUCTION` op; 3 op types + 2 data types; param struct; attr 148 | 13.1 | :577, :593-605, :640-648, :967 |
| Graph identity | 5 functions (`cuGraphNodeGetContainingGraph`, `…GetLocalId`, `…GetToolsId`, `cuGraphGetId`, `cuGraphExecGetId`) | 13.1 | :21502-21589 |
| Graph params | `cuGraphNodeGetParams` | 13.2 | :23063 |
| Recapture | 1 function; status enum and callback typedef; error 918 | 13.3 | :16691, :16629-16631, :3356 |
| CIG capture | 2 functions (`cuStreamBegin/EndCaptureToCig`); attr 151 | 13.2 | :16220, :16247, :968 |
| Graph misc | `CU_GRAPH_NODE_TYPE_RESERVED_16`; `CHILD_GRAPH_OWNERSHIP_INVALID`; func attr 16 | 13.3 (func attr) | :2052, :4927, :1176 |
| Launch attributes | attrs 17 and 18, plus 2 enums | 13.2 | :2194-2208, :2378-2381 |
| Memcpy | `cuMemcpyWithAttributesAsync`, `cuMemcpy3DWithAttributesAsync` | 13.2 | :11504, :11529 |
| Memory misc | `CU_MEM_LOCATION_TYPE_INVISIBLE`; 6 `CU_MEMPOOL_ATTR_*`; attr 152 plus its mmap doc | 13.2 (invisible), 13.3 (mmap) | :4314, :4642-4681, :969, :12899-12905 |
| Function introspection | `cuKernelGetParamCount`, `cuFuncGetParamCount` | 13.2 | :8747, :18704 |
| JIT | option 35 | 13.1 | :1629 |
| Green contexts / resources | 2 functions; `CU_GREEN_CTX_NONE`; SM group flags DEFAULT/BACKFILL; workqueue config (1000) and workqueue (10000) types; 2 scopes; `CUdevResource` layout change (`_oversize` 48 → 40 plus `nextResource`; size unchanged by arithmetic [inferred]) | 13.1 / 13.3 (`NONE`) | :26506-26613, :26981, :27250 |
| Multicast / fabric | 2 `_v2` bind functions; 12 `cuLogicalEndpoint*`; attrs 153-156 | 13.1 / 13.3 | :14392, :14511, :14664-15205, :970-973 |
| Coredump | 4 callback functions; 4 flags | 13.2 | :26348-26417, :25949-25952 |
| Errors | `STREAM_DETACHED` 917; `GRAPH_RECAPTURE_FAILURE` 918 | 13.1 / 13.3 | :3351, :3356 |
| Arrays | 14 YUV `CU_AD_FORMAT_*` | — | :775-788 |

**`cuda_runtime_api.h`.** Functions 310 → 337: **+27, −0**. Where each group starts:
- `cudaStreamBeginRecaptureToGraph` :2557
- `cudaFuncGetParamCount` :4189
- `cudaLaunchHostFunc_v2` :4322
- `cudaMemcpyWithAttributesAsync` :6620, plus a 3D form
- graph IDs, including `cudaGraphNodeGetToolsId` :11521
- `cudaGraphNodeGetParams` :13281
- `cudaGraphConditionalHandleCreate_v2` :13385 (adds a ctx parameter)
- `cudaDeviceGetDevResource` :14092
- `cudaDevSmResourceSplit` :14278
- `cudaGreenCtxCreate` :14369
- the `cudaExecutionCtx*` family, starting with `cudaExecutionCtxStreamCreate` :14547
- `cudaStreamGetDevResource` :14610

Green and execution contexts are new to the runtime API in this window.

**`driver_types.h`.** 60 new enum members:
- `cudaErrorVersionTranslation` = 10 (:268)
- `cudaErrorStreamDetached` and `cudaErrorGraphRecaptureFailure` (:1127, :1131)
- `cudaHostTaskBlocking` and `cudaHostTaskSpinWait` (:1535-1536)
- `cudaSharedMemoryMode*` (:1685-1688)
- `cudaDevAttrAtomicReductionSupported` = 148 (:2141)
- mempool attributes (:2217-2255)
- device-resource and workqueue enums (:3123-3160)
- `cudaKernelFunctionType` (:3531-3534)
- `cudaLaunchAttributeSharedMemoryMode` (:4178)

Struct fields added: `syncMode` on host node params, `deviceNodeUpdateStatus` on `cudaFuncAttributes`, and `ctx` on the memcpy, memset and kernel node params.

**`include/` directory.** 13.3 adds `cuda_tf32.h`, `cuda_tile.h`, `cufft_device.h`, `cuobjclient.h` and `cuobjtelem.h`, and drops `generated_cudart_removed_meta.h` [read].

## 4. PTX side

**ISA per toolkit.** 13.0 → 9.0, 13.1 → 9.1, 13.2 → 9.2, 13.3 → 9.3 (release notes, raw) [read]. 13.4 → 9.4 (13.4 release notes via WebFetch, plus the 9.4 PDF from the 13.4 docs) [read]. On the box, ptxas 13.0 accepts `.version` up to 9.0 and ptxas 13.3 up to 9.3; above that both reject with "Unsupported .version …; current version is …" [read].

**New instructions that apply to sm_86.**
- `clmad` (9.3, sm_80+)
- `.pragma "mma_throughput"` (9.3, all targets)
- `.language` (9.3)
- `ld/st .volatile .local` (9.1)
- `ld.acquire`/`st.release` with `.mmio` (9.3, sm_75+)

**New instructions that do not apply to sm_86.**
- `.u8x4`/`.s8x4` add/sub/min/max/neg (9.2): sm_107f and sm_120f families only.
- 9.4 packed `set`, mixed f16x2/bf16x2/f32x2 add and fma, `cvt .pzo`: sm_107f.
- prefetch `.valid_addr` and `ld .proxy::readonly`: sm_90+.
- `ldmatrix .s8/.s4`: sm_90a/sm_100f+.

Source: ptx_isa_9.3.pdf and ptx_isa_9.4.pdf via pdftotext [read].

**What our emitter writes.**
- LLVM 21.1.8 `llc -march=nvptx64 -mattr=help` lists `ptx32` … `ptx88` and CPUs up to sm_121f, so the newest ISA it can write is 8.8 [read].
- cuda-oxide @ b9847e9 adds `-mattr=+ptxNN` only when a kernel's requirement exceeds the recorded floor (`crates/cuda-oxide-codegen/src/ptx.rs:739-746`) [read].
- The sm_86 floor is 71 in `RECORDED_PTX_FLOORS`, which the crate says come "from the pinned LLVM 23 NVPTX backend… LLVM 21 does not accept every target recorded here". `PTX_ISA_SPELLINGS` stops at 90 (`crates/cuda-target-spec/src/lib.rs`) [read].
- Every module extracted from `generate`, `gate_p8` and `gate_e2e` is `.version 7.1 .target sm_86` [read].

**Driver JIT acceptance of 9.3/9.4: untested**, because it needs a context and the GPUs are held. R615 is CUDA 13.4, so I expect 9.4 [inferred]. Strings in the JIT library prove nothing either way. Test for when a card is idle: `cuModuleLoadData` of a two-line `.version 9.3` and `9.4` `.target sm_86` stub on each card.

**Can a toolkit switch change our kernels?** Not through the PTX path we use.
- The PTX comes from LLVM 21, which is independent of the toolkit.
- SASS comes from the driver's JIT (`cuda-host/src/embedded.rs:161-215`: a PTX payload goes to `load_module_from_image`) [read].
- The only toolkit input is libdevice, linked at IR level at build time. It is found in this order: `CUDA_OXIDE_LIBDEVICE` → `CUDA_TOOLKIT_PATH` → `CUDA_HOME` → `CUDA_PATH` → `/usr/local/cuda` → `/opt/cuda` (libnvvm-sys `find_libdevice`) [read].
  - md5: 13.0 is `6f7db61b…`, 13.3 is `7a29f14c…`.
  - A normalized per-function comparison found 19 of 352 functions changed: `cospi`, `cyl_bessel_i0f`, `cyl_bessel_i1f`, `erff`, `expm1f`, `fmodf`, `lgamma`, `lgammaf`, `llround`, `llroundf`, `mul24`, `nextafterf`, `powf`, `remainderf`, `remquof`, `sincospi`, `sinpi`, `tgamma`, `tgammaf`.
  - This is a lower bound, given the masking method. None of these is called by our kernels, which call `exp`, `sqrt`, `round` and `ex2.approx` [read].
  - That matches the lead's byte-identical `.oxart` (md5 `61edb10b…`).
- Two ways a toolkit *could* reach our kernels:
  - A future kernel calls a changed libdevice function.
  - A payload switches to NVVM IR or LTOIR. `generate` contains dlopen candidates `libnvJitLink.so.13/.12/.so` and libNVVM loader code (`strings`) [read]; it links no CUDA library (`ldd`) [read].
- The toolkit does move **ptx-scan**, whose ptxas reports ±1 register on 4 of 52 entries (F15). That is the instrument, not what runs.

## 5. ik baseline

**Today** [read]:
- ik `c10fbbcc` at `/home/user/ik_llama.cpp`.
- `build/CMakeCache.txt`: `CMAKE_CUDA_ARCHITECTURES=86`; compiler `/usr/local/cuda/bin/nvcc` (13.0, V13.0.88); all force flags OFF.
- `ggml/src/libggml.so` carries 228 sm_86 cubins plus 228 sm_86 PTX. RUNPATH is `/usr/local/cuda-13.0/targets/x86_64-linux/lib`, and `LD_LIBRARY_PATH` is empty.

What follows from that:
- ik runs ptxas-13.0 SASS; the driver picks the matching cubin over the PTX [inferred].
- It loads cudart and cuBLAS 13.0 despite the new `/etc/ld.so.conf.d/gds-13-3.conf`, because RUNPATH is searched before the ld.so cache [inferred].
- Nothing about ik changed tonight [inferred].

**If rebuilt with 13.3** (release-note facts only):
- **Device code.** ptxas 13.3 regenerates the 228 cubins.
  - 13.3 fixes a miscompile "present since CUDA 12.8": compiler-inserted reconvergence failing in kernels with two or more nested divergence levels. The 13.0 build is inside that window. Whether any ik kernel hit it is unknown.
  - 13.3 Math fixes `__mul24` with compile-time-constant inputs (a bug since 11.1).
  - libdevice 13.2 changed `expm1f` and `erff` ("minor accuracy improvements").
  - CCCL goes 3.0.1 → 3.3.4.
- **cuBLAS 13.0 → 13.6.0.2.**
  - INT8→INT32 `cublasLtMatmul` with N > 65,536 or batch > 1 was fixed in 13.1.
  - Green-context support arrived in 13.3.
  - Two known issues introduced in 13.3 are still listed in 13.3.1:
    - GEMV-like operations (M or N = 1) may return `NOT_SUPPORTED` "for certain shapes and workspace configurations".
    - `cublasLtMatmulAlgoGetHeuristic()` while capturing a non-default blocking stream returns `CUBLAS_STATUS_INTERNAL_ERROR`. The 13.3.1 wording is right; the 13.3.0 page says `CUDNN_STATUS_INTERNAL_ERROR`, a typo.
  - No cuBLAS bullet from 13.0 through 13.3U1 names Ampere or compute capability 8.
- **Where ik uses cuBLAS.** Quantized matmuls go through MMQ/mmvq (`ggml/src/ggml-cuda/mmq.cu:178`, `ggml_cuda_should_use_mmq`). cuBLAS is called at `ggml-cuda.cu:1771, 1809, 1853, 1888, 2430, 2472` for non-quantized paths.
- **Not checked:** whether ik's decode on this model issues any cuBLAS call with N = 1, or a heuristic query while capturing. Either would put it in reach of those two known issues.

## 6. cutile-rs

**Architecture floor.**
- The cutile-rs compatibility table puts sm_86 at Tile IR 13.2 minimum and says "CUDA 13.3 is recommended" (README and compatibility.md via WebFetch).
- The Tile IR "Spec 13.2" changelog: "Added support for Ampere (sm_80, sm_86, sm_87, sm_88) and Ada (sm_89) architectures" (raw).
- The 13.3 stability page lists "Ampere sm_86 A40, RTX 3090 — Tile IR 13.2" (raw).
- The CUDA **13.3** toolkit notes announce "Added support for all Ampere and later architectures, sm_80 and later" (raw).
- The 13.2 toolkit notes have no CUDA Tile architecture section (read absence).
- The 13.1 notes say "The Tile-IR AS compiler currently supports only Blackwell-class devices" (raw).
- On the box, `tileiras` 13.3 lists sm_86 in `--gpu-name`, and `--list-versions` prints 13.1, 13.2, 13.3 [read].

Net: we now have a working pair, 13.3 `tileiras` with 13.2+ bytecode. A 13.2 toolkit is unverified for sm_86, and we do not need one.

**What 13.3 adds that works on sm_86** (compatibility table plus 13.3 notes) [read]:
- `alloca`, `pack`/`unpack`
- `make_strided_view` and `make_gather_scatter_view`; `load_view_tko`/`store_view_tko` now take 1-D index tensors
- `atomic_red_view_tko`
- the `i4` type
- `exp` `rounding_mode`
- the `num_worker_warps_per_cta` hint

Not on sm_86: `mmaf_scaled` (sm_100+), f8 (Hopper+ per the hardware matrix), f4E2M1FN (Blackwell), PDL (13.4 and sm_90+), `mmaf fast_acc` (Hopper FP8 only).

**Integer support.**
- The Tile IR hardware matrix lists i1, i8, i16, i32 and i64 as supported on Ampere (raw).
- i4 is a storage type: "i4 tiles must be converted to a supported integer type before they are used in operations" (13.3 notes, raw).
- cutile-rs binds `mmai`, "Integer Tensor-Core matrix multiply-accumulate", `since="V13_2"`, at `cutile/src/_core.rs:2537-2559` [read].
- MMA ops (`mmaf`, `mmaf_scaled`, `mmai`) are exempt from bit-identity guarantees. "If a configured tile shape is too small or does not align with a supported TC shape, the compiler may emit a non-TC implementation (typically FFMA on the CUDA cores)" (raw). Tile IR also "does not guarantee bit‑identical numerical results across different toolchain versions" (raw).
- An i8 `mmai` accumulating in i32 is exact in any order [inferred], so that caveat bites the float paths only.

**dp4a.** `gh api search/code q='dp4a repo:NVlabs/cutile-rs'` returned `total_count` 0. That is a search result over the default-branch index, not proof of absence. The 13.1–13.3 Tile release notes I read name no 4-way dot-product op.

**Other.** Bytecode "can be… JIT-compiled by the driver at load time" (stability page, raw); which Tile IR version R615's JIT accepts is untested. CHANGELOG 0.3.1 (2026-09-02): launch overhead 3.9 → 1.6 µs (WebFetch).

## 7. Risks and removals

**7.1 The instruments moved: the headline risk.**

**(a) ncu.** `/usr/local/cuda-13.0/bin/ncu` is a wrapper that runs the newest `/opt/nvidia/nsight-compute/*`. 2026.2.1 was installed Sep 23 02:14, so `tools/ref/ncu-gpu.sh:25` now runs it [read]. Checked against the runner:
- `--clock-control` now defaults to boost. The runner passes `base` (:120), so no change [read].
- `--graph-profiling` still defaults to node, and the runner passes node [read].
- All 16 metric names the runner requests exist in both versions' `--query-metrics --chip ga102`: 13 `smsp__warp_issue_stalled_*_per_warp_active`, plus the L1 and L2 hit rates and DRAM bytes. The list grew from 4625 to 4663 lines [read].
- Every section label in `KEEP` exists in both versions [read].
- **But "Local Memory Spilling Requests" changed meaning.** It is now `sass__inst_executed_register_spilling_mem_local`; it was `sass__inst_executed_register_spilling` (`MemoryWorkloadAnalysis.section:21-22`, 2026.2.1 vs 2025.3.1). Shared-memory spills moved to a new "Shared Memory Spilling Requests" row that `KEEP` does not read [read].
  - A kernel that spills to shared memory would show 0 in the kept column [inferred].
  - ptxas reports 0 spills for our entries, but ncu measures the driver-JIT SASS, so "no numeric change for us" is [inferred], not measured.
  - This column is load-bearing: the runner ties it to the past 13–31× spill defect (nvlabs-ledger §5).
- WarpStateStatistics adds a not-issued warp-ID sample group, and Occupancy drops the cluster chart (CC_90+ only) [read].
- `[config]` (:90) prints the ncu version, so every record says which instrument produced it [read].

**(b) Upstream candidate (NVIDIA Nsight Compute, via the forum or bug channel; not the NVlabs ledger).**
- In 2026.2.1, `derived__local_spilling_requests_pct` = 100·`mem_local` / (`mem_local_op_read` + `mem_local_op_write`), and `mem_local` is defined as that same sum (`2026.2.1/sections/MemoryWorkloadAnalysis.section:13-14, 30`) [read].
- So "Local Memory Spilling Request Overhead" reads 100 % whenever any local spill exists, and 0 otherwise [inferred, algebra on read expressions].
- In 2025.3.1 the denominator was all local loads and stores.
- This comes from code reading only. FAIL-first would need a kernel with local spills profiled under both versions.

**(c) nsys.**
- `/usr/local/cuda-13.0/bin/nsys` is a wrapper pinned to 2025.3.2; 2026.1.3 is installed next to it [read].
- From 2025.6, export row order is no longer guaranteed, "including timestamp order", and 2026.1 removed text and JSON export [read]. `nsys-gpu.sh` exports sqlite and sorts by start (:77, :83), so it is unaffected [read].
- B2 needs host-function trace, which starts in 2025.6 [read], so that round must call 2026.1.3 explicitly.

**(d) ptx-scan.** `PTX_SCAN_PTXAS` defaults to `/usr/local/cuda/bin/ptxas`, i.e. 13.0 (`tools/ptx-scan.sh:42`) [read]. If a base tree and a changed tree are scanned with different ptxas versions, the "move, split, rename" proof (identical tables) fails for a reason that is not the change [inferred].

**7.2 Everything that follows `/usr/local/cuda` moves together.** It stays at 13.0 on purpose. Following it are: the nsys wrapper, ptx-scan's ptxas, cuda-oxide's libdevice fallback, the bindgen input for cuda-bindings (the lead saw the bindings differ under 13.3), and ik's `nvcc`. The ncu wrapper also lives there, though it already picks the newest install on its own. Repointing the link moves all of them at once [inferred].

**7.3 Global loader path.** The 13.3 install added `/etc/ld.so.conf.d/gds-13-3.conf` [read]. ik is shielded by its RUNPATH [read]. `generate` loads nvJitLink and libNVVM only for NVVM-IR/LTOIR payloads, and which toolkit's copy it would get depends on the ld.so cache order [inferred].

**7.4 Bindings.** 13.3 removes no driver function and keeps the 47 remaps identical, so the entry points we call resolve the same [read]:
- `cuStreamBeginCapture_v2` (`crates/gpu/src/graph.rs:75`)
- `cuGraphGetNodes` (:114)
- `cuGraphInstantiateWithFlags` (:125)
- `cuGraphLaunch` (:140)

The structs that changed (`CUDA_HOST_NODE_PARAMS_v2`, `CUdevResource`; `CUgraphNodeParams` stays at 232 B) are unused today [read].

**7.5 Removals in 13.1–13.3 that touch us: none found** [read].
- 13.3 "Deprecated Architectures: None".
- The 13.1 and 13.2 deprecation sections repeat 13.0 items.
- 13.3 drops the Nsight Eclipse plugins.
- `generated_cudart_removed_meta.h` is gone.
- nvcc `--sanitize` became `--fdevice-sanitize` (memcheck only).

**7.6 Correctness.** F19 applies to ik's 13.0-built SASS window. The `__mul24` fix touches a function we do not call; libdevice's `mul24` changed `mul nsw` → `mul` [read].

**7.7 Green contexts, if we ever use them.** The header gives no concurrency or forward-progress guarantee [read]. Kernels that size grids to the device's SM count, as our grid-barrier kernels do, must use the green context's count instead. cuSOLVER shipped exactly that hang, fixed per the 13.4 notes [read/inferred].

**7.8 Open items for the lead.**
1. When a card is free: query attributes 122, 123, 148 and 152 on both cards (`cuDeviceGetAttribute` needs no context), and make one `cuStreamWaitValue32`/`WriteValue32` call.
2. Driver-JIT stub load test for `.version` 9.3 and 9.4.
3. `gdscheck -p`, and the PCIe topology (NVMe vs A6000 root complex), before any GDS round.
4. Pin ncu to 2025.3.1, or adopt 2026.2.1 as a new table.
5. Remove `~/repo/bloomery-cudares` on the box (`just box-gc`, `just box-tracks --remove`).

## 8. Improvement spots outside this task

Report only; I touched none of these.
- `tools/ref/ncu-gpu.sh:25`: `NCU` resolves through a wrapper that picks the newest install. Pin a versioned path, or fail on an unexpected version. About 3 lines.
- `tools/ref/ncu-gpu.sh:16-18`: the comment credits node-level output to "ncu 2025.3의 기본값". The runner passes `--graph-profiling node` itself (:120), and the tool is now 2026.2.1. 1 line.
- `tools/ref/ncu-gpu.sh`, `KEEP` tuple in the summary heredoc: add "Shared Memory Spilling Requests" so the spill diagnostic stays whole under 2026.2.1. 1 line.
- `tools/ref/nsys-gpu.sh:28`: same wrapper pattern; today it runs 2025.3.2, which has no host-function trace for B2. 1 line.
- `tools/ref/nsys-gpu.sh:83`: rows are ordered by `start` and cut at replay boundaries. The export already carries `graphNodeId` (621 distinct values over 637,767 rows in `/root/bloomery-data/nsys/nsys-d1024-graph-234404.sqlite`), which could key rows to nodes directly. About 10 lines.
- `tools/ptx-scan.sh:42`: print `ptxas --version` in the table header, so tables from two ptxas versions are never compared. About 2 lines.
- Build env (`bloomery-env.sh` on the box): set `CUDA_OXIDE_LIBDEVICE` explicitly, so the one toolkit input to our kernels is named rather than found through `/usr/local/cuda`. 1 line.
- `docs/RESULTS-mul30-gpu-scout.md:15, :73`: "cutile-rs sm_8x 요건(13.2)" is the bytecode floor. The toolkit that announces Ampere support is 13.3 (now installed), and a 13.2 toolkit is unverified for sm_86. 1–2 lines.
- The brief, not the tree: it cites the cuda-oxide checkout `b0f961d`, but the pin is `b9847e9` (`Cargo.toml:9-10`; box `/root/.cargo/cuda-oxide/source-rev.txt`). `b0f961d` is a second checkout in the cargo git cache.

## 9. Sources consulted, tools, box footprint

**Box, read-only.**
- Both toolkits under `/usr/local/cuda-13.{0,3}/`: `version.json`; the three headers above; `include/` listings; `nvvm/libdevice/libdevice.10.bc`; `bin/nvcc`, `ptxas` (help and version probes); `tileiras`; the `bin/ncu` and `bin/nsys` wrappers.
- `/opt/nvidia/nsight-compute/{2025.3.1,2026.2.1}`: `sections/*.section`, `--help`, `--query-metrics --chip ga102`.
- Nsight Systems 2025.3.2 and 2026.1.3: `--help`.
- CUPTI headers from both toolkits.
- Driver: `libcuda.so.615.71.09` (`nm -D`); `libnvidia-ptxjitcompiler.so.615.71.09` (`ls`, `strings`).
- System: `/etc/ld.so.conf.d/`; `/proc/driver/nvidia/params`; one `nvidia-smi -q`; `uname -r`; `/boot/config-*`; `df -T /models`.
- LLVM: `~/opt/LLVM-21.1.8-Linux-X64/bin/llc -mattr=help`.
- cuda-oxide: `/root/src/cuda-oxide` and `/root/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/{b9847e9,b0f961d}` (target-spec, codegen `ptx.rs`, libnvvm-sys, cuda-host `embedded.rs`/`lib.rs`).
- Our binaries: `~/repo/bloomery/target/release/{generate,gate_p8,gate_e2e}` (PTX extraction, `ldd`, `strings`).
- ik: `/home/user/ik_llama.cpp` (`CMakeCache.txt`, `libggml.so`, `ggml-cuda.cu`, `mmq.cu`).
- The nsys sqlite export named in §8.

**Mac, read-only.**
- `docs/plan.md`, `docs/arch-split.md`, `docs/RESULTS-mul30-gpu-scout.md`
- `tools/box.sh`, `tools/ref/ncu-gpu.sh`, `tools/ref/nsys-gpu.sh`, `tools/ptx-scan.sh`
- `crates/gpu/src/{graph,elem,router,q5,cores,flash}.rs`, `crates/q3k-gemv/src/main.rs`
- `Cargo.toml`, `Cargo.lock`

**Web.**
- Toolkit release notes: `docs.nvidia.com/cuda/archive/{13.1.0,13.2.0,13.3.0,13.3.1}/cuda-toolkit-release-notes/`, plus the current page (13.4 U1).
- PTX ISA: `docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf`, `…/developer-preview/13.4/pdf/ptx_isa_9.4.pdf`.
- Driver API archive pages `…/archive/<v>/cuda-driver-api/group__CUDA__{TYPES,MEM,LIBRARY,STREAM,COREDUMP,MULTICAST,GREEN__CONTEXTS,EXEC,GRAPH,MEMOP}.html`.
- GDS: `docs.nvidia.com/gpudirect-storage/{release-notes,best-practices-guide}/`.
- Nsight Compute: `docs.nvidia.com/nsight-compute/ReleaseNotes/`.
- Nsight Systems: `archive.docs.nvidia.com/nsight-systems/{2025.5,2025.6,2026.1}/ReleaseNotes/`.
- Tile IR: `docs.nvidia.com/cuda/tile-ir/{13.1/sections/appendix.html, 13.2/sections/release_notes.html, 13.3/sections/{stability,release_notes}.html}`.
- cutile-rs on GitHub: README, `CHANGELOG.md`, `cutile-book/reference/compatibility.md`, `cutile/src/_core.rs`.
- cuda-python issue #2359; CUPTI `structCUpti__ActivityKernel13`.

**Tools.**
- Bash: curl plus a python tag strip for raw quotes; pdftotext for the ISA PDFs; `gh api search/code` for the dp4a search.
- WebFetch, which paraphrases. Rows sourced only through it are labelled "(WebFetch)".
- `tools/box.sh` with `BLOOMERY_REMOTE='~/repo/bloomery-cudares'` for the box.
- The GitHub MCP server failed to connect (HTTP 400, bad Authorization header) and was not used.

**Box footprint.**
- Every box.sh call rsynced the tree into `~/repo/bloomery-cudares`, which now exists and needs cleanup.
- Temporary files `/tmp/q-{2025.3.1,2026.2.1}.txt` and `/tmp/cudares-q-{2025.3.1,2026.2.1}.txt` were created and removed; `ls` confirmed they are gone.
- One `nvidia-smi -q` query.
- Everything else was offline and read-only: `ncu --query-metrics --chip`, `diff`, `grep`, `strings`, `nm`, `ldd`, `ptxas` on stdin.
- No GPU job, no context creation, no installs, no edits outside `/tmp`.

## 10. Model

Ran as Claude Opus 5.5 (`claude-opus-5-5[1m]`), spawned as opus.
