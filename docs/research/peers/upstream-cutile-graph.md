# Pre-Submission Research Report: CUDA Graph Wrapper Upstream Candidate (`cuda-core` / `NVlabs/cutile-rs`)

**Target Repository**: `https://github.com/NVlabs/cutile-rs` (crate `cuda-core`)  
**Secondary Repository**: `https://github.com/NVlabs/cuda-oxide`  
**Downstream Consumer**: `/Users/hckim/repo/bloomery` (`crates/gpu/src/graph.rs`, `crates/gpu/src/lib.rs`, `crates/gpu-gates/src/bin/gate_p2.rs`)  
**Investigation Commit Hashes**:
- `NVlabs/cutile-rs`: `e04245bdcf1f5bfc602a2078168eff90ee40bebb` (2026-09-18)
- `NVlabs/cuda-oxide`: `b0f961df3af0ff140b3b006fa2b6750b71f43f62` (2026-09-20)
- `NVlabs/cutile-rs` Tag `v0.3.1`: `cdc69c13a7529552a26d9941893f047970d8e95f` (2026-09-04)

---

## 1. State at HEAD

### 1.1 Where `cuda-core` and `cuda-bindings` live
- **Read in code**:
  - `cuda-core`: `cutile-rs/cuda-core/` (member of root workspace `cutile-rs/Cargo.toml:9`, commit `e04245bdcf1f5bfc602a2078168eff90ee40bebb`).
  - `cuda-bindings`: `cutile-rs/cuda-bindings/` (member of root workspace `cutile-rs/Cargo.toml:6`, commit `e04245bdcf1f5bfc602a2078168eff90ee40bebb`).

### 1.2 Version at HEAD and `0.3.1` Tag Status
- **Read in code**:
  - In `cutile-rs/Cargo.toml:53`, the workspace package version at HEAD is `0.4.0` (`version = "0.4.0"`).
  - `cuda-core/Cargo.toml:version.workspace = true` and `cuda-bindings/Cargo.toml:version.workspace = true`, so both crates are at `0.4.0` at HEAD.
  - `0.3.1` is an annotated git tag named `v0.3.1` (ref `refs/tags/v0.3.1`, object `2e4510f5af577b35797a159c91ef3b1fe1235847`), pointing to commit `cdc69c13a7529552a26d9941893f047970d8e95f` tagged by Melih Elibol on 2026-09-04T04:04:19Z. The tag has the `v` prefix (`v0.3.1`).

### 1.3 Graph Capture, Instantiate, and Launch Occurrences in `cuda-core` at HEAD
Every occurrence of graph capture, instantiate, and launch in `cuda-core` at commit `e04245bdcf1f5bfc602a2078168eff90ee40bebb` was verified via repository search:

- **Capture** (found only in legacy cudarc shim and legacy `runtime::Stream`):
  1. `cuda-core/src/cudarc_shim.rs:324-333`:
     ```rust
     /// Begins stream capture for graph construction.
     ///
     /// # Safety
     /// `stream` must be valid and not already capturing.
     pub unsafe fn begin_capture(
         stream: cuda_bindings::CUstream,
         mode: cuda_bindings::CUstreamCaptureMode,
     ) -> Result<(), DriverError> {
         cuda_bindings::cuStreamBeginCapture_v2(stream, mode).result()
     }
     ```
  2. `cuda-core/src/cudarc_shim.rs:335-345`:
     ```rust
     /// Ends stream capture and returns the captured graph.
     ///
     /// # Safety
     /// `stream` must be in a capturing state.
     pub unsafe fn end_capture(
         stream: cuda_bindings::CUstream,
     ) -> Result<cuda_bindings::CUgraph, DriverError> {
         let mut graph = MaybeUninit::uninit();
         cuda_bindings::cuStreamEndCapture(stream, graph.as_mut_ptr()).result()?;
         Ok(graph.assume_init())
     }
     ```
  3. `cuda-core/src/cudarc_shim.rs:347-357`:
     ```rust
     /// Queries whether the stream is currently capturing.
     ///
     /// # Safety
     /// `stream` must be a valid stream handle.
     pub unsafe fn is_capturing(
         stream: cuda_bindings::CUstream,
     ) -> Result<cuda_bindings::CUstreamCaptureStatus, DriverError> {
         let mut status = MaybeUninit::uninit();
         cuda_bindings::cuStreamIsCapturing(stream, status.as_mut_ptr()).result()?;
         Ok(status.assume_init())
     }
     ```
  4. `cuda-core/src/runtime.rs:902-912`:
     ```rust
     /// Begins stream capture for CUDA graph construction.
     ///
     /// # Safety
     /// The caller must ensure the context is current and the stream is not
     /// already being captured.
     pub unsafe fn begin_capture(
         &self,
         mode: cuda_bindings::CUstreamCaptureMode,
     ) -> Result<(), DriverError> {
         stream::begin_capture(self.cu_stream, mode)
     }
     ```
  5. `cuda-core/src/runtime.rs:914-920`:
     ```rust
     /// Ends stream capture and returns the captured CUDA graph.
     ///
     /// # Safety
     /// The caller must ensure `begin_capture` was previously called on this stream.
     pub unsafe fn end_capture(&self) -> Result<cuda_bindings::CUgraph, DriverError> {
         stream::end_capture(self.cu_stream)
     }
     ```
  6. `cuda-core/src/simt/context.rs:175`:
     Textual comment match only: `/// query-only, all three concern CIG (graphics-interop) contexts, and none`.

- **Instantiate**:
  - **Read in code**: Exactly **zero** occurrences in `cuda-core`. No `cuGraphInstantiate` or `cuGraphInstantiateWithFlags` exists anywhere in `cuda-core`.

- **Launch**:
  - **Read in code**: Exactly **zero** occurrences in `cuda-core`. No `cuGraphLaunch` exists anywhere in `cuda-core`.

### 1.4 Does `simt::CudaStream` have capture methods?
- **Read in code**: **NO**. In `cuda-core/src/simt/stream.rs:1-253`, `CudaStream` defines only:
  - `cu_stream(&self) -> cuda_bindings::CUstream` (line 74)
  - `context(&self) -> &Arc<CudaContext>` (line 79)
  - `priority(&self) -> Result<i32, DriverError>` (line 91)
  - `synchronize(&self) -> Result<(), DriverError>` (line 102)
  - `query(&self) -> Result<bool, DriverError>` (line 128)
  - `fork(&self) -> Result<Arc<Self>, DriverError>` (line 143)
  - `join(&self, other: &CudaStream) -> Result<(), DriverError>` (line 169)
  - `record_event(&self, flags: Option<cuda_bindings::CUevent_flags>) -> Result<CudaEvent, DriverError>` (line 178)
  - `wait(&self, event: &CudaEvent) -> Result<(), DriverError>` (line 192)
  - `launch_host_function<F: FnOnce() + Send>(&self, host_func: F) -> Result<(), DriverError>` (line 218)
  It has **no** `begin_capture`, **no** `end_capture`, and **no** `is_capturing` methods.

### 1.5 Is there any `Graph`-like type in `cuda-core`?
- **Read in code**: **NO**. `cuda-core` defines no `Graph` struct or graph abstraction.
- **Read in code**: In the sibling crate `cuda-async` within the same repository (`cutile-rs/cuda-async/src/cuda_graph.rs:45`), there is a `pub struct CudaGraph<T>`. However, `CudaGraph<T>` is built specifically for `cuda-async`'s `DeviceOp` trait, requires an `Arc<cuda_core::Stream>` (the legacy `runtime::Stream`, NOT `simt::CudaStream`), and uses an async execution lock (`acquire_execution_lock`).

### 1.6 Check `NVlabs/cuda-oxide` at HEAD for graph support
- **Read in code** (commit `b0f961df3af0ff140b3b006fa2b6750b71f43f62`):
  - `grep -rnI -i "cugraph"`: **0 occurrences** across the entire repository.
  - `grep -rnI -i "capture"`: Occurs only in compiler AST/closure capture code (e.g. `crates/mir-importer/src/translator/rvalue/aggregate.rs:667`), LLVM intrinsic metadata (`NoCapture` in `intrinsics/catalog.json`), or subprocess `capture_output`.
  - `cuda-oxide` ships **no CUDA graph wrapper** and **no graph example**.
  - `cuda-oxide` launchers generated by `#[cuda_module]` take `&::cuda_core::CudaStream` (`cuda-oxide/crates/cuda-macros/src/cuda_module/launchers.rs:262, 324, 348`).

---

## 2. Duplicates (Search Log)

Queries executed across issues and PRs (both open and closed) via GitHub Search API, plus repository Discussions:

### 2.1 Query Hit Counts

| Query String | `NVlabs/cutile-rs` Issues | `NVlabs/cutile-rs` PRs | `NVlabs/cuda-oxide` Issues | `NVlabs/cuda-oxide` PRs |
|:---|:---:|:---:|:---:|:---:|
| `graph` | 5 | 15 | 22 | 44 |
| `cuda graph` | 5 | 13 | 16 | 36 |
| `capture` | 2 | 10 | 12 | 40 |
| `cuGraph` | 0 | 1 | 0 | 2 |
| `begin_capture` | 0 | 0 | 0 | 2 |
| `replay` | 1 | 7 | 3 | 12 |
| `instantiate` | 1 | 1 | 6 | 11 |

**Discussions**:
- `NVlabs/cutile-rs`: 6 total discussions.
  - Discussion #80: *"v0.0.2 API preview: DeviceOp, tensor references, and CUDA graph scope"* ([https://github.com/NVlabs/cutile-rs/discussions/80](https://github.com/NVlabs/cutile-rs/discussions/80)).
- `NVlabs/cuda-oxide`: 8 total discussions. None relate to CUDA graphs.

### 2.2 Direct Citations of Key Relevant Issues and PRs

#### A. In `NVlabs/cuda-oxide`:
1. **Issue #107** [OPEN]: *"CUDA Graph API"*  
   - URL: [https://github.com/NVlabs/cuda-oxide/issues/107](https://github.com/NVlabs/cuda-oxide/issues/107)  
   - Opened by `@mtrsl`.  
   - **Stated by upstream** (`@nihalpasham`, maintainer, on 2026-09-03):  
     > *"Short answer: yes, and it arrives through the shared host crates, not oxide's own. oxide's cuda-bindings / cuda-core / cuda-async are being swapped for the crates cutile-rs publishes (#1102, draft, CI builds pass). That is why #346 and #585 were closed rather than merged: a graph layer in the old crates would be deleted with them. The shared cuda-async already has a graph module built on stream capture... What is left on my side: oxide's launchers implement a different trait (same job) from the one record accepts today, so the shared crate needs a small bridge for them, plus an oxide example with buffers allocated outside the capture."*
2. **PR #574** [CLOSED]: *"feat(cuda-core): RAII CUDA graph capture and replay"*  
   - URL: [https://github.com/NVlabs/cuda-oxide/pull/574](https://github.com/NVlabs/cuda-oxide/pull/574)  
   - Author: `@honeyspoon`. Proposed RAII graph capture in `cuda-core`. Closed in favor of #585.
3. **PR #585** [CLOSED]: *"feat(cuda-core): safe CUDA graph capture and replay"*  
   - URL: [https://github.com/NVlabs/cuda-oxide/pull/585](https://github.com/NVlabs/cuda-oxide/pull/585)  
   - Author: `@honeyspoon`. Replaced #574 to make buffer lifetimes unrepresentable in safe code.  
   - **Stated by upstream** (`@nihalpasham`, 2026-07-30):  
     > *"I'm closing this for direction, not quality: same reasoning as #346: we're moving cuda-oxide's host-side crates (cuda-bindings/cuda-core/cuda-async) onto the crates.io-published crates cutile-rs uses, so a Graph API added to cuda-core today lands on a foundation we're about to swap out. This design belongs upstream in those crates, where the current graph surface is the raw begin/end_capture + CUgraph FFI your PR body describes. #107 tracks how that unfolds."*
4. **PR #346** [CLOSED]: *"feat: fp and cuda-graph support"*  
   - URL: [https://github.com/NVlabs/cuda-oxide/pull/346](https://github.com/NVlabs/cuda-oxide/pull/346)  
   - Closed for direction during the host-crate migration.

#### B. In `NVlabs/cutile-rs`:
1. **PR #270** [OPEN]: *"fix: lifetime-brand CudaGraph so captured buffers cannot dangle or race"*  
   - URL: [https://github.com/NVlabs/cutile-rs/pull/270](https://github.com/NVlabs/cutile-rs/pull/270)  
   - Author: `@elibol` (lead maintainer).  
   - **Read in code**: Redesigns `CudaGraph<T>` to `CudaGraph<'a, T>`, mutably borrowing captured buffers for its lifetime so dropping or mutating a buffer while replayable is a compile-time error. Changes `launch()` to take `&mut self` and return `GraphLaunch<'_, 'a, T>`.
2. **PR #254** [OPEN] & **PR #253** [CLOSED/MERGED]: *"feat: add CUDA graph capture modes"*  
   - URL #254: [https://github.com/NVlabs/cutile-rs/pull/254](https://github.com/NVlabs/cutile-rs/pull/254)  
   - URL #253: [https://github.com/NVlabs/cutile-rs/pull/253](https://github.com/NVlabs/cutile-rs/pull/253)  
   - Author: `@honeyspoon`. Adds `CaptureMode::{Global, ThreadLocal, Relaxed}`.
3. **PR #199** [CLOSED/MERGED]: *"feat(cuda-async): expose the raw CUgraph and CUgraphExec handles"*  
   - URL: [https://github.com/NVlabs/cutile-rs/pull/199](https://github.com/NVlabs/cutile-rs/pull/199)  
   - Author: `@honeyspoon`. Merged by `@elibol`. Exposes raw handles for dot-print and inspection.
4. **PR #200** [CLOSED/MERGED]: *"feat: inspect what stream capture actually recorded"*  
   - URL: [https://github.com/NVlabs/cutile-rs/pull/200](https://github.com/NVlabs/cutile-rs/pull/200)  
   - Author: `@honeyspoon`. Merged by `@elibol`. Exposes recorded node inspection.
5. **Issue #3** [CLOSED]: *"Cuda Graph Example"*  
   - URL: [https://github.com/NVlabs/cutile-rs/issues/3](https://github.com/NVlabs/cutile-rs/issues/3)  
   - Opened by `@jafioti` asking how to use cutile kernels in graphs for LLaMA forward pass (~75ms overhead bottleneck).
6. **Issue #69** [CLOSED]: *"Cudarc based frameworks interop"*  
   - URL: [https://github.com/NVlabs/cutile-rs/issues/69](https://github.com/NVlabs/cutile-rs/issues/69)  
   - Deep discussion between `@OlivierDehaene` (Hugging Face Candle maintainer) and `@elibol`. `@OlivierDehaene` questioned the async overhead and `cuda-async` coloring; `@elibol` committed to keeping `cuda-core` minimal with `borrow_raw` constructors, while placing the high-level safe graph and op abstractions in `cuda-async`.

---

## 3. Contribution Rules

### 3.1 CONTRIBUTING and Process
- **Read in code** (`cutile-rs/CONTRIBUTING.md:1-71`):
  - Commits require cryptographic signature (SSH or GPG).
  - Commits must carry a DCO sign-off line: `git commit -s` generating `Signed-off-by: Your Name <your@email.com>`.
  - Branch naming: `<type>/<description>` (e.g. `feat/warp-interop`).
  - PR title format: `<type>: <description>` in lowercase (e.g. `feat: add warp interop support`).
  - Validation: `bash scripts/run_all.sh` (or `scripts/run_cpu_tests.sh` on non-GPU machines).

### 3.2 CLA / DCO
- **Read in code** (`cutile-rs/CONTRIBUTING.md:50-70`):
  - Developer Certificate of Origin (DCO) Version 1.1 is enforced.
  - There is **no corporate CLA** form or click-through agreement beyond the git commit sign-off.

### 3.3 AI-Content Policy
- **Read in code**: Neither `NVlabs/cutile-rs` nor `NVlabs/cuda-oxide` specifies any AI-generated code policy. There is no mention of LLMs, GitHub Copilot, or AI tools in `CONTRIBUTING.md`, issue/PR templates, or repository rules.

### 3.4 External PR Acceptance (Evidence of Non-NVIDIA Merged PRs)
External PRs are actively merged. Five verified merged PRs by non-NVIDIA authors:
1. **PR #291** ([https://github.com/NVlabs/cutile-rs/pull/291](https://github.com/NVlabs/cutile-rs/pull/291)): *"doc: fix data parallel MLP tutorial"* by `@notbowen` (Hu Bowen, `contact@hubowen.dev`). Signed-off.
2. **PR #282** ([https://github.com/NVlabs/cutile-rs/pull/282](https://github.com/NVlabs/cutile-rs/pull/282)): *"fix: expose kernel cache eviction without autotuning"* by `@arcusbuilds` (Srijan Keshri, `srijankeshri007@gmail.com`). Signed-off.
3. **PR #277** ([https://github.com/NVlabs/cutile-rs/pull/277](https://github.com/NVlabs/cutile-rs/pull/277)): *"fix(cuda-bindings): add WSL2 and multiarch driver paths to dynamic loader"* by `@emersonbusson` (Emerson Busson, `emersonbusson@gmail.com`).
4. **PR #241** ([https://github.com/NVlabs/cutile-rs/pull/241](https://github.com/NVlabs/cutile-rs/pull/241)): *"fix: version persisted tuning cache keys"* by `@almightychang` (Sam, Joochul Chang, `almightychang@icloud.com`, RLWRLD, Inc.). Signed-off.
5. **PR #204** ([https://github.com/NVlabs/cutile-rs/pull/204](https://github.com/NVlabs/cutile-rs/pull/204)): *"fix: adapt CUDA driver flag types"* by `@Heltion` (Heltion, `heltion@qq.com`). Signed-off.
*(Additional external merges: PR #190 by `@svenstaro` / Arch Linux, PR #188 by `@ivarflakstad`, PR #183 by `@Firestar99` / Vectorware Inc, PR #169 by `@fallintoplace`).*

### 3.5 Release Cadence of `cuda-core` on crates.io
- **Read from crates.io API**: 9 published versions:
  - `v0.0.0-alpha`: 2026-03-13
  - `v0.0.1-alpha`: 2026-03-14
  - `v0.0.1`: 2026-04-14
  - `v0.0.2`: 2026-04-27 (2 weeks)
  - `v0.1.0`: 2026-05-16 (3 weeks)
  - `v0.1.1`: 2026-06-02 (2 weeks)
  - `v0.2.0`: 2026-06-16 (2 weeks)
  - `v0.3.0`: 2026-08-21 (~2 months)
  - `v0.3.1`: 2026-09-04 (2 weeks)
- **Inference**: Releases happen frequently, roughly every 2–4 weeks (with minor feature gaps up to 2 months). The workspace is currently bumped to `0.4.0` in git, indicating a 0.4.0 release is pending.

---

## 4. API Fit

### 4.1 How `cuda-core` Wraps Driver Objects
Examining `cuda-core/src/simt/`:
- **Ownership**:
  - `CudaStream` (`simt/stream.rs:40-45`), `CudaEvent` (`simt/event.rs:31-36`), `CudaModule` (`simt/module.rs:41-47`), and `DeviceBuffer` (`simt/device_buffer.rs:142-154`) each hold `ctx: Arc<CudaContext>`. The context outlives the object.
- **Context Binding**:
  - Every method calls `self.ctx.bind_to_thread()?` before issuing driver FFI (`simt/context.rs:382`).
- **Error Type**:
  - Methods return `Result<T, DriverError>`, wrapping the raw `cudaError_enum`.
- **Drop Conventions**:
  - In `Drop`, types do not panic and do not silently ignore errors. They call:
    ```rust
    self.ctx.record_err(self.ctx.bind_to_thread());
    self.ctx.record_err(unsafe { cuda_bindings::cu...Destroy(self.handle).result() });
    ```
    Errors are recorded onto the `CudaContext` via `record_err`.
- **Thread Safety**:
  - Driver handles implement `unsafe impl Send` and `unsafe impl Sync` with documentation that handles can be enqueued/queried from any thread as long as the context is bound.
- **Naming Conventions**:
  - Concrete driver wrappers use the `Cuda*` prefix (`CudaContext`, `CudaStream`, `CudaEvent`, `CudaModule`, `CudaFunction`). A graph wrapper in `simt` would follow this convention as `CudaGraph`.

### 4.2 Comparison with Bloomery Downstream Wrapper (`bloomery/crates/gpu/src/graph.rs`)

| Dimension | Downstream `bloomery::Graph` | Upstream `cuda-core` Conventions | Upstream `cuda-async::CudaGraph` (at HEAD & PR #270) |
|:---|:---|:---|:---|
| **Context Ownership** | None. Holds raw `CUgraph`, `CUgraphExec`, `nodes: usize`. | Requires `ctx: Arc<CudaContext>`. | Holds `Arc<Stream>` and `Arc<Device>`. |
| **Error Handling** | Maps to custom `GpuError` via string format (`cu()`). | Returns `Result<T, DriverError>`. | Returns `Result<T, DeviceError>`. |
| **Drop Handling** | Silent `unsafe { cuGraphExecDestroy; cuGraphDestroy; }`. | Calls `ctx.bind_to_thread()` and `ctx.record_err()`. | Binds device and calls `sys::cuGraph...Destroy(handle).result()`. |
| **Stream Association** | Stream passed by `&CudaStream` per call (`capture`, `launch`). | Objects retain their owning context or stream. | Owns `Arc<Stream>` and pre-stages exec via `cuGraphUpload`. |
| **Send / Sync** | Implements `Send`, but NOT `Sync`. | Implements `Send + Sync` where driver is thread-safe. | Implements `Send + Sync` for `GraphExecHandle`. |
| **Replay Mutability** | `launch(&self, stream: &CudaStream)`. | N/A (no wrapper in `simt`). | `launch(&mut self)` (PR #270) to prevent concurrent replay. |
| **Buffer Lifetimes** | **Unbranded** (`Graph` has `'static` lifetime). | N/A | **Lifetime-branded** (`CudaGraph<'a, T>`), borrowing buffers (PR #270). |

### 4.3 What Makes Replay Unsafe (and why a safe wrapper is non-trivial)
1. **Pointer Stability**:
   - **Read in code**: CUDA graph nodes record raw device virtual addresses (`CUdeviceptr`) at capture time. If any `DeviceBuffer` used during capture is deallocated, reallocated at a different address, or moved before `launch`, the graph replays against invalid or reallocated VRAM (use-after-free or memory corruption).
   - In bloomery, this safety is maintained *by construction* via `GpuModel` resident buffers (allocated once at load), but cannot be guaranteed by the Rust type system without lifetime branding.
2. **Frozen Launch Scalars**:
   - Kernel parameters passed by value (e.g. dimensions `m`, strides, flags) are copied into driver parameter memory during capture. Replay does **not** re-read host variables. Re-launching with changed host scalars will silently replay the old scalar values unless graph node parameter updates (`cuGraphExecKernelNodeSetParams`) are explicitly called.
3. **Contract / Precondition Bypass**:
   - `PreparedLaunch` validates launch contracts (`requires` clauses, shape bounds, cooperative limits) on the host when building kernel parameters. On graph replay, host launch code is completely bypassed; only the driver executes the captured node DAG.
4. **Panic / Error Handling During Capture**:
   - In bloomery `graph.rs:87-93`, if the captured closure panics, `cuStreamEndCapture` is skipped. The stream remains permanently in capture mode (`CU_STREAM_CAPTURE_STATUS_INVALIDATED`), stranding the stream and corrupting subsequent work. A robust wrapper must use `std::panic::catch_unwind` (as `cuda-async` does in `capture_on`) to ensure `cuStreamEndCapture` runs on all unwind paths.
5. **Concurrent Replay**:
   - Replaying a single `CUgraphExec` concurrently on multiple threads or overlapping on different streams is illegal in the CUDA driver. Taking `&self` in `launch(&self)` allows concurrent launches in safe code; taking `&mut self` enforces exclusive replay at compile time.

---

## 5. Issue-First or PR-First?

### 5.1 Upstream Practice and Precedents
- **Stated by upstream**:
  1. In `cuda-oxide#585`, maintainer `@nihalpasham` closed a complete PR for safe CUDA graph capture in `cuda-core` because NVIDIA is standardizing graph support in `cuda-async` on top of `DeviceOp`.
  2. In `cuda-oxide#107`, `@nihalpasham` explicitly stated that graph replay is designed to live in `cuda-async` (`CudaGraph::scope`), and what is pending is bridging `cuda-oxide`'s module launchers into `cuda-async`'s `DeviceOp`/`record` interface.
  3. In `cutile-rs#270`, lead maintainer `@elibol` is actively landing a major overhaul of `cuda-async::CudaGraph` to add lifetime branding (`CudaGraph<'a, T>`).
- **Inference**:
  Submitting a PR with an imperatively captured `Graph` struct directly to `cuda-core` will almost certainly be closed for architectural direction, as it contradicts the maintainers' published design plan of keeping high-level abstractions in `cuda-async`.

### 5.2 The Smallest Single-Purpose Change: PR Candidate vs Issue
- **Smallest PR Candidate**:
  Add `begin_capture`, `end_capture`, and `is_capturing` to `simt::CudaStream` in `cuda-core/src/simt/stream.rs`:
  ```rust
  impl CudaStream {
      pub unsafe fn begin_capture(&self, mode: cuda_bindings::CUstreamCaptureMode) -> Result<(), DriverError> {
          self.ctx.bind_to_thread()?;
          cuda_bindings::cuStreamBeginCapture_v2(self.cu_stream, mode).result()
      }

      pub unsafe fn end_capture(&self) -> Result<cuda_bindings::CUgraph, DriverError> {
          self.ctx.bind_to_thread()?;
          let mut graph = MaybeUninit::uninit();
          cuda_bindings::cuStreamEndCapture(self.cu_stream, graph.as_mut_ptr()).result()?;
          Ok(graph.assume_init())
      }

      pub unsafe fn is_capturing(&self) -> Result<cuda_bindings::CUstreamCaptureStatus, DriverError> {
          self.ctx.bind_to_thread()?;
          let mut status = MaybeUninit::uninit();
          cuda_bindings::cuStreamIsCapturing(self.cu_stream, status.as_mut_ptr()).result()?;
          Ok(status.assume_init())
      }
  }
  ```
  - **Why this works**:
    1. The legacy `runtime::Stream` in `cuda-core/src/runtime.rs:907, 918` already has these exact methods.
    2. `simt::CudaStream` omitted them purely during the initial `simt` fork from `cuda-oxide`.
    3. It is unopinionated, requires zero new types, adds zero lifetime entanglement, and unblocks downstream crates (like bloomery) from needing to call raw bindgen FFI `cuStreamBeginCapture_v2(stream.cu_stream(), ...)`.
- **Recommendation**:
  **Issue-first** referencing `cuda-oxide#107` and `cutile-rs#270`, or a targeted PR implementing *only* the low-level stream capture methods on `simt::CudaStream`.

---

## 6. Adversarial Review

What an upstream maintainer (`elibol` or `nihalpasham`) would object to:

1. **API Surface & Architectural Placement**:
   - *Maintainer Objection*: *"We do not want graph types in `cuda-core`. `cuda-core` provides minimal, unopinionated driver bindings and context-bound primitives. Higher-level graph abstractions belong in `cuda-async` (`cuda_async::cuda_graph::CudaGraph`), where we are already integrating `DeviceOp` and lifetime branding (#270). If you want graph replay, you should use `cuda_async`."*
   - *Counter-argument for the minimal PR*: Adding `begin_capture`/`end_capture` directly to `simt::CudaStream` does not add high-level abstractions; it merely restores parity with `runtime::Stream`.
2. **Driver-Version Floor for `cuGraph*`**:
   - *Toolkit / Driver Introduced*:
     - `cuGraphCreate`, `cuGraphDestroy`, `cuGraphInstantiate`, `cuGraphLaunch`, `cuStreamBeginCapture`, `cuStreamEndCapture` were introduced in **CUDA 10.0** (September 2018; NVIDIA CUDA Toolkit 10.0 `<cuda.h>`).
     - `cuStreamBeginCapture_v2` was introduced in **CUDA 10.1**.
     - `cuGraphInstantiateWithFlags` was introduced in **CUDA 11.0**.
   - *Upstream Floor*: `cuda-bindings/build.rs:21` sets `const MIN_CUDA_VERSION: u32 = 13000;` (CUDA 13.0). The CUDA graph driver API is well below upstream's minimum supported toolkit floor.
3. **Untestable in CI Without a GPU**:
   - *Read in code*: `cutile-rs/.github/workflows/pr.yml:22` runs on `linux-amd64-cpu16` inside standard containers without GPU hardware.
   - Any test that invokes `cuStreamBeginCapture` or `cuGraphLaunch` will fail if executed on CPU runners.
   - Upstream tests in `cuda-async/tests/cuda_graph.rs:15-19` handle this by guarding all execution behind `if !has_gpu() { return; }`. Any PR adding graph methods must follow this convention, or the test suite in CI will break.
4. **Direct Overlap with Planned Features**:
   - Issue `cuda-oxide#107` specifically tracks adding graph support for `cuda-oxide` launchers via the shared host crates.
   - PR `cutile-rs#270` is already in flight refactoring `CudaGraph`.
   - Maintainers are actively drafting `cuda-oxide#1102` to swap host crates to `cutile-rs`.

---

## Deliverables Summary

### (a) Verdict: Go / No-Go / Needs-More
- **Full `Graph` Wrapper in `cuda-core`**: **NO-GO**.
  - **Strongest Reason**: Upstream already explicitly rejected this in `cuda-oxide#585` and is actively implementing their preferred safe graph architecture in `cuda-async` (PR #270 and issue #107). Submitting a parallel `Graph` type to `cuda-core` would be rejected for direction.
- **Low-Level Capture on `simt::CudaStream`**: **GO** (as a minimal PR).
  - **Strongest Reason**: Exposing `begin_capture`, `end_capture`, and `is_capturing` on `simt::CudaStream` completes the missing driver methods on the simt stream (bringing it into parity with `runtime::Stream`), does not introduce competing abstractions, and requires fewer than 35 lines of code.

### (b) What Could Not Be Determined
1. **Private CI Testing**: Whether NVIDIA maintains an internal, hardware-backed GPU test pipeline that mirrors PRs before merge, or if maintainers rely exclusively on manual local runs (`scripts/run_all.sh`) on workstation hardware.
2. **Release Date of 0.4.0**: When PR #270 and PR #1102 are targeted to land and be tagged on crates.io.

### (c) Additional Contribution Opportunities in `cuda-core` for High-Frequency Launch Users
Items noticed in `cuda-core` that impact applications driving many small kernel launches per step (e.g. LLM decode steps):

1. `cuda-core/src/simt/context.rs:382-392`: `bind_to_thread()` issues `cuCtxGetCurrent` FFI on every launch/stream call; caching the active thread-local context would eliminate a driver round-trip per kernel launch.
2. `cuda-core/src/simt/launch.rs:906-932`: `PreparedLaunch::prepare()` runs 8 separate driver attribute queries on every invocation; caching static function/device limits on `CudaFunction` / `CudaContext` would eliminate redundant queries on dynamic shapes.
3. `cuda-core/src/simt/device_buffer.rs:171`: `DeviceBuffer::drop()` for asynchronous allocations calls `self.ctx.synchronize()`, forcing a full device context stall on normal RAII drops.
4. `cuda-core/src/simt/stream.rs:218-231`: `launch_host_function()` allocates a heap `Box` per callback (`Box::new(host_func)`), imposing host allocator overhead on high-frequency callback pipelines.

---

## Report Formats & Verifications

### 1. Files Changed
- `<scratch>/up-cutile/report.md`: Created pre-submission investigation report.
- `<scratch>/up-cutile/search_gh.py`: Created helper script to synchronously query GitHub search API.
- `<scratch>/up-cutile/search_results.json`: Cached search results from GitHub.

### 2. Completion Criteria Commands with Real Output
Commands used during verification:
```
$ gh api repos/NVlabs/cutile-rs/commits/main --jq '.sha, .commit.message, .commit.committer.date'
e04245bdcf1f5bfc602a2078168eff90ee40bebb
fix: DGX Spark (aarch64, sm_121) bring-up: candle behind a feature, isolated lifetime tests, aarch64 lints (#295)

Signed-off-by: Melih Elibol <elibol@users.noreply.github.com>
2026-09-18T17:45:26Z

$ gh api repos/NVlabs/cuda-oxide/commits/main --jq '.sha, .commit.message, .commit.committer.date'
b0f961df3af0ff140b3b006fa2b6750b71f43f62
fix(tests): use the upstream volatility interface

Update the control-flow regression to query StoreOp through VolatilityOpInterface after the upstream volatility migration. Include vecadd and abi_hmm in the tracked rust-analyzer project settings.

Signed-off-by: nihalpasham <nihalp@nvidia.com>
2026-09-20T10:52:57Z
```

### 3. Self-Verification
1. **Existing behavior this change removed or weakened**: None; research-only task with no repository modifications.
2. **New constants/mappings/tables**: None added to codebase.
3. **Other surfaces**: Not applicable (investigation only).
4. **Changed test assertions**: None.
5. **Contract clauses that conflicted**: None.

### 4. What Could Not Be Implemented or Verified
- Internal NVIDIA GPU CI runner configuration (public repository only contains CPU runner definitions).

### 5. What Was Deliberately Left Untouched
- `cutile-rs` and `cuda-oxide` cloned trees in `scratchpad/up-cutile/src/`: Left unmodified in compliance with read-only repository rules.
- Bloomery codebase (`/Users/hckim/repo/bloomery`): Left untouched as instructed.

### 6. Improvement Opportunities Noticed Beyond Spec
- `cutile-rs/cuda-core/src/simt/context.rs:382`: Avoid calling `cuCtxGetCurrent` on every launch by caching the last-bound context handle per thread (size: ~1 round / design question).
- `cutile-rs/cuda-core/src/simt/device_buffer.rs:171`: Provide an async-safe drop queue or pool to avoid synchronizing the whole context when dropping `DeviceBuffer`s (size: ~1 round).
