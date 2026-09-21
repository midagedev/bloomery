# FeLLM vs bloomery: Comparative Investigation of CUDA-Oxide GGUF Quantized Decode

**Author**: Antigravity  
**Date**: 2026-09-21  
**Target Hardware / Model Context**: DeepSeek-V2-Lite Q3_K_M, batch-1 decode, NVIDIA GeForce RTX 3090 (sm_86), single captured CUDA graph per step, bit-identical rerun discipline.  
**Repositories Investigated**:
- **Peer**: FeLLM (`plugins/cuda_kernels`, commit `bdd6507cff168e5a74236f35d381682ec592e903`)
- **Ours**: bloomery (`bloomery/`, `crates/gpu`, `docs/gpu-design.md`)

---

## 1. Q4_K / Q6_K GEMV Side-by-Side

### 1.1 Architecture & Geometry

| Dimension | FeLLM (`plugins/cuda_kernels/src/oxide_kernels.rs`) | bloomery (`crates/gpu/src/lib.rs`, `cores.rs`) |
| :--- | :--- | :--- |
| **Q4_K Kernel Entry** | `q4k_q8_gemv_warp` (L1414), `q4k_q8_gemv_multiwarp` (L1492), `q4k_q8_gemv_warp4` (L1304) | `q4k_gemv` (L237), `cores::q4k_a_chain` (`cores.rs:108`) |
| **Q4_K Block / Grid Geometry** | `q4k_q8_gemv_warp`: 1 output row per block. Block = 4 warps (128 threads) if `n_blocks < 32`, 8 warps (256 threads) if `n_blocks >= 32` (`launchers.rs:58`). `multiwarp`: 4 rows per block, 4 warps (`launchers.rs:66`). | 1 warp (32 threads) per output row. Block = 256 threads (8 warps) computing 8 output rows concurrently (`lib.rs:225, 248`). |
| **Q4_K Warp Partitioning** | In `_warp`, warps split super-blocks along K: `block = warp_id; block += nwarps` (L1440, 1457). Inside each warp, 32 lanes cover 128 elements per half (`j = half * 128 + lane * 4`), so each lane owns 4 elements per half (L1445). | Warp covers 4 super-blocks per iteration (`sbp = 4*it + (lane>>3)`). Lane `s = lane & 7` owns the **entire 32-value sub-block `s`** throughout the iteration (`lib.rs:261, 276`). |
| **Q6_K Kernel Entry** | `q6k_gemv_row` (L679), `q6k_q8_gemv_multiwarp` (L1095), `q6k_q8_gemv_warp4` (L987) | `q6k_gemv` (L504), `cores::q6k_chain` (`cores.rs:257`) |
| **Q6_K Default Dispatch** | **`q6k_gemv_row`** (`launchers.rs:1003`): **1 single thread per output row**! (`cfg_1d(rows * out_dim)`). | 1 warp (32 threads) per output row. Block = 256 threads (8 warps) computing 8 output rows (`lib.rs:493, 514`). |
| **Weight Access Pattern** | Raw unaligned GGUF byte buffer `w: &[u8]`. Reads `d, dm` via byte slice indexing and `f16_to_f32` (L1223). Unaligned `load_u32` for packed nibbles (L1172, 1185). Byte-by-byte scalar reads for scales/mins (L1198, 1211). In Q6_K, scalar byte reads `w[ql + ql_idx]` and `w[qh + lane]` (L955, 957). | Repacked `w: &[u32]`. Aligned u32 loads (`lib.rs:287-293`). In Q6_K, odd 210-byte super-blocks (2 mod 4) are loaded as aligned 5-word windows and shifted via **hardware 16-bit funnel shifter (`funnel16`)** in registers (`lib.rs:571, 617`). |
| **Activation Format** | `Q8_32`: `qx: &[i8]`, `x_scales: &[f32]` (32 values per scale block, L1306, 1417). | `q8_1`: 128-value blocks (`lib.rs:70-113`). Permuted slots so 32 lanes read 32 contiguous words (coalesced 128B L1 line). Precomputed `s8: &[i32]` signed-byte sums per 32-value group. |
| **Int8 Dot Formulation** | `dp4a_s32(qp, xp, 0) + 8 * sx` (L1245). Q6_K in `_multiwarp` uses `dp4a_s32` on manually assembled u32 words (L972, 981). Default Q6_K (`q6k_gemv_row`) uses scalar f32 multiplies! | Chained integer `dp4a_s32` in registers: `q4k_a_chain` (8 chained dp4a, `cores.rs:124-131`), `q6k_chain` (4 chained dp4a, `cores.rs:258-261`). |
| **Scale / Min Application** | Applied on **every 4-element chunk** inside `q4k_q8_chunk_pre` (L1247): `xs * (d * scale * dot - dm * min * sx)`. | Hoisted to **32-value sub-block**: `cda = d*sc`, `cdb = 8*cda - dmin*mi` (`cores.rs:160`). Dot is `(a*cda + b*cdb)*e0` where `b` is a single `s8` integer load. |
| **Reduction Tree** | Warp shuffle reduction (5 steps) -> store to shared memory `WARP_SUMS[warp_id]` -> `thread::sync_threads()` barrier -> Warp 0 shuffle reduction (5 steps) -> Lane 0 store (L1459-1486). | Pure register warp shuffle reduction `warp::reduce_sum_f32(f0)` (5 steps) -> Lane 0 store (`lib.rs:424-430`). **Zero shared memory, zero block barriers**. |

---

### 1.2 Operation Counts for 1 Output Row (K = 2048, 8 Super-Blocks, Batch-1 / $M=1$)

#### FeLLM (`q4k_q8_gemv_warp`, 4 warps = 128 threads)
- **Memory Loads Issued by Block**:
  - Weight loads: Each thread processes 2 super-blocks (4 halves). Each super-block loads `d, dm` (2 f16s = 4 bytes) by every thread ($128 \times 2 = 256$ loads). In each half, thread executes `q4k_signed4` (1 `load_u32`), `q4k_scale` (1 byte load), `q4k_min` (1 byte load) $\rightarrow 3$ loads $\times 4$ halves $\times 128$ threads $= 1,536$ loads. Total weight load instructions: **1,792 loads**.
  - Activation loads: In each half, each thread loads `xp` (1 `load_u32`) and `xs` (1 f32 load) $\rightarrow 2$ loads $\times 4$ halves $\times 128$ threads $= \mathbf{1,024}$ **activation loads**.
  - Shared memory traffic: 4 stores to `WARP_SUMS`, 4 loads from `WARP_SUMS`.
- **Integer Multiply-Adds**:
  - 512 `dp4a_s32` instructions ($128 \text{ threads} \times 4 \text{ halves}$, computing $512 \times 4 = 2,048$ signed int8 multiply-adds).
- **Float Operations**:
  - In `q4k_q8_chunk_pre`: each 4-element chunk performs `d * sc * dot` (2 muls), `dm * mi * sx` (2 muls), subtraction (1 sub), `* xs` (1 mul), accumulator add (1 add) $= 7$ float ops.
  - Across 512 chunks: $512 \times 7 = 3,584$ float ops.
  - Reductions: 4 warps $\times 5$ shuffle adds ($20$) + Warp 0 reduction ($4 \text{ adds} + 5 \text{ shuffle adds} = 9$) $= 29$ float ops.
  - **Total Float Ops**: **~3,613 ops per row**.

#### bloomery (`q4k_gemv`, 1 warp = 32 threads, 2 iterations)
- **Memory Loads Issued by Warp**:
  - Weight loads: Each lane in each iteration loads header words `w0..w3` (4 u32 loads) and 8 qs words (8 u32 loads) $\rightarrow 12$ loads $\times 2 \text{ iters} \times 32 \text{ lanes} = \mathbf{768}$ **aligned u32 loads**.
  - Activation loads: `q4k_a_chain` loads 8 u32 words from `q` (coalesced 128B L1 lines across 32 lanes), 1 load from `s8` (coalesced 128B line), 1 load from `d8` $\rightarrow 10$ loads $\times 2 \text{ iters} \times 32 \text{ lanes} = \mathbf{640}$ **loads** (issuing only 16 coalesced 128B line transactions for `q`, 2 for `s8`, 2 for `d8`).
  - Shared memory traffic: **0 bytes** (pure registers).
- **Integer Multiply-Adds**:
  - 512 `dp4a_s32` instructions (32 lanes $\times$ 8 dp4a in `q4k_a_chain` $\times$ 2 iters $= 2,048$ signed int8 multiply-adds).
- **Float Operations**:
  - Per lane iteration (32-element sub-block): `q4k_coeff` (1 mul, 1 FMA), inner accumulation `(a*cda + b*cdb)*e0` (2 muls, 1 add, 1 mul), `f0 += ...` (1 add) $= 7$ float ops.
  - Across 32 lanes $\times$ 2 iterations: $64 \times 7 = 448$ float ops.
  - Warp reduction: 5 shuffle adds.
  - **Total Float Ops**: **~453 ops per row** (**~8.0x fewer float operations than FeLLM**).

---

### 1.3 Speed & Quantization Noise Verdict

1. **Speed on sm_86 (RTX 3090)** (*Inference*):
   - **bloomery is substantially faster**:
     - **Float Op Intensity**: FeLLM executes 3,613 float ops/row vs bloomery's 453 float ops/row because FeLLM recomputes scale/min float conversions and multiplications on every 4-element chunk, whereas bloomery hoists them to 32-element sub-blocks and precomputes the activation sum (`s8`) in integer domain.
     - **Occupancy & Synchronization**: bloomery packs 8 independent rows per block (256 threads), with zero barriers and zero shared memory. FeLLM assigns 1 row per block of 128 threads, requiring `thread::sync_threads()` and shared memory staging between warps.
     - **Memory Coalescing**: bloomery loads aligned u32 words and uses permutation mapping to guarantee 128-byte L1 transaction coalescing. FeLLM issues scattered unaligned byte and word reads.
     - **Q6_K Default**: FeLLM's `launchers.rs:1003` dispatches Q6_K to `q6k_gemv_row`—a **single-threaded scalar loop** with zero warp parallelism and zero dp4a. bloomery runs full warp-cooperative funnel-shifted GEMV.
2. **Activation Quantization Noise** (*Read in code & measured in `docs/gpu-design.md:26`*):
   - **FeLLM has lower quantization noise**: FeLLM uses `Q8_32` (32-value blocks with an f32 scale per 32 values), whereas bloomery uses `q8_1` with 128-value blocks.
   - In bloomery's `docs/gpu-design.md:26`, real activation profiling on RTX 3090 against `ref_cuda` revealed that 128-value quantization exhibits 1.3x–2.0x higher noise than ik CUDA's 32-value blocks (reaching up to $2.39 \times 10^{-2}$ relative error on high-variance norm outputs where `amax/rms` is 17–41). FeLLM's 32-value quantization avoids this degradation.

---

## 2. Dispatch Logic for the Three Variants (`warp4`, `warp`, `multiwarp`)

### 2.1 Dispatch Code in FeLLM
- **Location**: `plugins/cuda_kernels/src/launchers.rs:74-76`, `L926-966`, `L1465-1485`, `L1607-1627`.
```rust
fn mmvq_multirow(out_dim: u32, n_blocks: u32) -> bool {
    n_blocks >= 16 && out_dim >= 4096
}
```

### 2.2 Variant Characteristics and Rationale

1. **`q4k_q8_gemv_multiwarp`** (`oxide_kernels.rs:1492`):
   - **Condition**: `n_blocks >= 16` ($K \ge 4096$) AND `out_dim >= 4096`.
   - **Geometry**: 4 output rows per block, 4 warps (128 threads). Grid: `(out_dim.div_ceil(4) * rows, 1, 1)`.
   - **Rationale**: Stated by author in doc comment (L1490): *"Decode MMVQ: four output rows share the Q8 activation."* When $K$ and $M$ are large, 4 rows can reuse the loaded Q8 activation chunk across 4 warps, reducing activation memory bandwidth while splitting $K$ across warps to cap latency.
2. **`q4k_q8_gemv_warp`** (`oxide_kernels.rs:1414`):
   - **Condition**: Default fallback when `mmvq_multirow` is false ($K < 4096$ or $\text{out\_dim} < 4096$).
   - **Geometry**: 1 output row per block. Block = 4 warps (if $K < 8192$) or 8 warps (if $K \ge 8192$).
   - **Rationale**: Stated by author in doc comment (L1411): *"One output row per block; eight warps split Q4_K super-blocks. Prefer this when `n_blocks` is modest so the GPU stays occupancy-bound."* For modest dimensions, dedicating an entire block to split a single row provides maximum occupancy across GPU SMs.
3. **`q4k_q8_gemv_warp4`** (`oxide_kernels.rs:1304`):
   - **Status in repo**: **Unused orphan** (*read in code*: never invoked in `launchers.rs` or anywhere in FeLLM).
   - **Geometry**: 1 warp (32 threads) computes 4 output rows sequentially along $K$ without multi-warp parallel reduction.
   - **Inference**: A preliminary prototype before multiwarp block-reduction was implemented; kept in `oxide_kernels.rs` but abandoned in the runtime launcher.

---

## 3. Fused Gate·Up·SwiGLU (`q4k_gate_up_swiglu_multiwarp`)

### 3.1 Kernel Implementation Details
- **Location**: `plugins/cuda_kernels/src/oxide_kernels.rs:1610-1687`
- **Launcher**: `plugins/cuda_kernels/src/launchers.rs:1654-1735`
- **Graph Construction**: `crates/fellm-model/src/graph.rs:2272-2287`
- **Lowering**: `crates/fellm-runtime/src/cuda_lowering.rs:163-170`

1. **What is Fused**:
   - Computes both Gate GEMV ($W_{\text{gate}} \cdot x$) and Up GEMV ($W_{\text{up}} \cdot x$) simultaneously, and evaluates the SwiGLU activation in the epilogue:
     $$\text{out}[r] = \text{silu}(\text{gate}_r) \cdot \text{up}_r = \frac{\text{gate}_r}{1 + e^{-\text{gate}_r}} \cdot \text{up}_r$$
2. **Geometry**:
   - Grid: `(out_dim, 1, 1)` (1 block per output row of the intermediate dimension).
   - Block: 128 threads (4 warps: `tid >> 5`).
   - 4 warps split the super-blocks along $K$: `block = warp_id; block += 4`.
   - In each inner loop step, each lane loads activation chunk `xp = load_i8x4(qx, xi)` **once**, and computes both:
     ```rust
     gate += q4k_q8_chunk(gate_w, gate_blk, j, xp, sx, xs);
     up   += q4k_q8_chunk(up_w,   up_blk,   j, xp, sx, xs);
     ```
   - Inter-warp reduction via `GATE_SUMS[warp_id]` and `UP_SUMS[warp_id]` in shared memory.
   - Warp 0 Lane 0 evaluates SwiGLU:
     ```rust
     let silu = gate_total / (1.0 + (-gate_total).exp());
     *out.get_unchecked_mut(row) = silu * up_total;
     ```
3. **Is Output Re-quantized in the Same Launch?**
   - **NO** (*read in code: `out: DisjointSlice<f32>` at L1617*). It writes unquantized `f32` intermediate values directly to global memory.

### 3.2 Comparison with bloomery's Planned 4-Launch Dense FFN
- **bloomery Plan** (`docs/gpu-design.md:62`):
  1. `[norm + 128-value quant]`
  2. `[gate·up·swiglu]`
  3. `[32-value quant]`
  4. `[down + residual]`
  Total: **4 launches**.
- **FeLLM Actual Execution Path**:
  1. `launch_rmsnorm`: separate launch, outputs `f32` (`lib.rs:372`).
  2. `quantize_q8_32`: separate launch, invoked inside launcher helper `with_q8_activation` (`launchers.rs:1704`).
  3. `q4k_gate_up_swiglu_multiwarp`: separate launch, outputs `f32` (`launchers.rs:1717`).
  4. `quantize_q8_32`: separate launch on intermediate SwiGLU buffer, invoked inside down projection (`launchers.rs:1020`).
  5. `q4k_q8_gemv_multiwarp` / `warp`: down projection with fused residual addition via `fuse_residual: u32` (`launchers.rs:1490`).
  Total: **5 launches**.
- **Key Difference**: FeLLM does **not** fuse `rms_norm` with activation quantization; it runs `rmsnorm` into an f32 buffer and launches `quantize_q8_32` as an auxiliary kernel. However, FeLLM **does** fuse residual addition into the down GEMV store (`total + skip` at `oxide_kernels.rs:1485`), which bloomery can adopt immediately.

---

## 4. MoE Implementation (`moe_q4k_project*`, `DeviceStepParams`)

### 4.1 Routing Location & Top-K Selection
- **Location**: `plugins/cuda_kernels/src/oxide_kernels.rs:2164-2260` (`moe_route_topk`), `launchers.rs:2120`.
- **Where it runs**: **Exclusively on DEVICE**.
  - Author comment (L2161): *"Router projection, activation, and top-k selection. One device thread owns a token so expert IDs never cross the host boundary."*
  - Single thread computes router dot products against all experts, computes softmax or sigmoid gating, selects top-k via iterative scan, and writes `ids: &[u32]` and `scores: &[f32]` into device scratch buffers.

### 4.2 Expert Addressing Inside Captured CUDA Graph
- **Mechanism**:
  - `ids` and `scores` are preallocated device scratch buffers whose pointers are invariant across graph replays.
  - In `moe_q4k_project` (L2342, 2360):
    ```rust
    let assignment = linear / out_dim as usize;
    let expert = ids[assignment] as usize;
    let value = dot_q4k(weights, expert * expert_bytes + (row_offset + row) * row_bytes, input, xb, blocks);
    ```
  - The kernel reads `ids[assignment]` dynamically on device. Because `ids` resides at a static address in the captured graph, replay executes correctly without host interaction.

### 4.3 Launches per MoE Layer & Output Combination
- **Total Launches per MoE Layer in FeLLM**: **6 launches** (*read in `launchers.rs:2118-2239`*):
  1. `moe_route_topk` (routing + top-k selection)
  2. `moe_q4k_project` (gate projection across all $k$ assignments)
  3. `moe_q4k_project` (up projection across all $k$ assignments)
  4. `silu_gate` (elementwise SwiGLU activation)
  5. `moe_q4k_project` or `moe_q6k_project` (down projection across all $k$ assignments)
  6. `moe_weighted_reduce` (weighted sum across top-k experts into `outd`)
- **Are Expert Outputs Combined in Same Launch?**
  - **NO**. `moe_weighted_reduce` (`oxide_kernels.rs:2507`) is a distinct 6th kernel launch.

### 4.4 Severe Performance Anti-Pattern in FeLLM's MoE
- In batch-1 decode (`tokens == 1`), `launchers.rs:2428-2435` dispatches to `moe_q4k_project`:
  - Grid: `cfg_1d(assignments * ff)`.
  - **Exactly 1 thread per output row** running `dot_q4k` (`oxide_kernels.rs:354`)!
  - `dot_q4k` is an unvectorized scalar loop performing unaligned byte reads from global memory and scalar f32 math without coalescing or `dp4a_s32`.
- **bloomery's Superior Alternative**:
  - `crates/gpu/src/lib.rs:1161` (`q3k_gemv_sel`) and `crates/gpu/src/q5.rs:553` (`q5_0_gemv_sel`):
  - Uses full warp-cooperative execution (1 warp per row, 8 warps per block), reading `sel[slot]` indirectly on device, with vectorized u32 loads and chained `dp4a_s32`.

---

## 5. Attention Decode (`attention_fa2/fa3_decode_paged`)

### 5.1 Geometry & Parallelism
- **FeLLM** (`oxide_kernels.rs:4887-5104` for FA3, `L5107-5230` for FA2):
  - Grid: `(n_heads, 1, 1)` $\rightarrow$ **Exactly 1 block per query head**.
  - Block: 128 threads (4 warps).
  - In FA3: Warp 0 is the producer warp (loads 16-key KV tiles into SMEM `K_BUF`, `V_BUF`). Warp 1 is the consumer warp (computes online softmax). Warps 2 and 3 are completely idle (*read in code: `warp == 1` check at L5020*).
  - **GPU SM Under-utilization**: For DeepSeek-V2-Lite with MLA ($n_{\text{heads}} = 16$), FeLLM launches only **16 blocks total on the entire GPU**! On an RTX 3090 with 82 SMs, 66 SMs (80.5%) sit completely idle during attention decode.
- **bloomery** (`crates/gpu/src/flash.rs:319-370`):
  - 1 warp per `(token, head)` query row. Block = 256 threads (8 warps) computing 8 query rows concurrently.
  - Full SM occupancy across the card.

### 5.2 How Sequence Length Reaches the Kernel in a Graph
- **FeLLM**:
  - Read directly from device-resident parameter struct:
    ```rust
    let seq = control_u32(params, 8); // oxide_kernels.rs:4952, 5153
    ```
  - Byte offset 8 corresponds to `DeviceStepParams.sequence_length` (`physical_plan.rs:111`).
  - The host writes 48 bytes via `update_step_params` before graph launch; the kernel reads it dynamically.
- **bloomery**:
  - Reads `n_keys_buf: &[u32]` passed as a device slice argument (`flash.rs:346`).

### 5.3 Online Softmax Reduction Order
- **FeLLM**:
  - Processes keys **strictly sequentially, 1 key at a time** in an outer loop over `t_local < tile_len` (L5027, 5186).
  - For each single key: reduces `q · k` across 32 lanes with `shuffle_down_f32`, broadcasts to lane 0, updates scalar `running_max` and `running_sum`, rescales accumulators, and accumulates $V$.
- **bloomery**:
  - Processes keys in **32-key block-parallel steps** (`flash.rs:139-218`).
  - All 32 lanes evaluate 32 keys concurrently. A 5-step butterfly `reduce_max_f32` and `reduce_sum_f32` updates the entire 32-key block at once, followed by 32 broadcast iterations for $V$.

### 5.4 KV Cache Layout & Depth 4096+ Scaling Verdict
- **KV Cache Handling**:
  - FeLLM: Paged cache (`block_table[layer * n_logical + logical]`). Loads f16 bytes through scalar unaligned reads (`*arena.get_unchecked(...)`), unpacks to f32 into SMEM tiles.
  - bloomery: Absorbed latent cache (`[rope_dims | latent]` u16), walks 64-value chunks with FMA, unpacking `half_to_f32` directly into registers without SMEM staging.
- **Scaling to Depth 4096+**:
  - **bloomery scales drastically better**:
    - At depth 4096, FeLLM's consumer warp executes **4,096 sequential iterations** of warp shuffle reductions ($4096 \times 5 = 20,480$ shuffles per head) while SMs are under-utilized.
    - bloomery executes $4096 / 32 = \mathbf{128}$ **block iterations**, amortizing softmax rescaling and achieving full occupancy across all SMs.

---

## 6. Graph Engine Architecture

### 6.1 Lifecycle & Replay
- **Location**: `crates/backend-cuda/src/graph.rs`, `crates/fellm-runtime/src/engine.rs:3196-3233`
- **Lifecycle**:
  1. Step 0: `full_step_warmed = false`. Executes uncaptured step (`step.run(...)`) to populate internal scratch buffers and initialize CUDA contexts.
  2. Step 1: `full_step_warmed = true` and `graph.is_none()`. Calls `cuda.begin_graph_capture()`, enqueues step operations (`step.enqueue(...)`), finishes capture with `cuGraphInstantiateWithFlags` and `cuGraphUpload`, storing `CudaGraphExec`.
  3. Step 2+: Replays via `graph.launch()?` (`cuGraphLaunch`).

### 6.2 Graph Count: One Single Graph
- FeLLM maintains **EXACTLY ONE GRAPH** (`decode.graph: Option<CudaGraphExec>` at `engine.rs:506`).
- **No sequence-length bucketing**: Because `sequence_length`, `position`, and KV write indices are read dynamically from `DeviceStepParams`, graph topology and buffer bindings never change across token positions.

### 6.3 Upload Before Replay & Host Overhead
- **Data Uploaded**: Exactly one 48-byte struct (`DeviceStepParams` at `physical_plan.rs:105`) uploaded via `cuda.update_step_params(&params)` (`backend.rs:151`).
- **Host Overhead per Step**:
  - FeLLM incurs unnecessary host overhead in `engine.rs:3165-3191`: it iterates over CPU bindings (`bindings.rope`, `bindings.kv_write`, `bindings.attention`) calling `step.set_attrs(...)` on every step, even though the captured graph ignores host attribute updates.
  - `graph.launch()?` issues a single `cuGraphLaunch` call.

---

## 7. Testing Discipline & Performance Claims

### 7.1 Assertions, Tolerances & References
- **`plugins/cuda_kernels`**: **ZERO automated tests** (*verified: no `#[test]` anywhere in crate*).
- **`crates/backend-cuda`**:
  - Only two unit tests in `plan.rs:444-463` testing static arena offset interval overlap logic.
  - Example smoke test `crates/backend-cuda/examples/graph_smoke.rs:80, 170`: tests live graph replay of `Add` (`[11.0, 22.0, 33.0, 44.0]`, exact) and `RoPE` (asserts `abs_diff < 1e-5`).
- **`crates/fellm-runtime`**:
  - `tests/attention_correctness.rs:75`: tests host CPU reference vs CPU FA2 (`err < 1e-4`).
- **Determinism / Bit-Identity**: **NONE**. FeLLM has no rerun bit-identity assertions or hash pinning.

### 7.2 Performance Claims in Repository
- **Stated by Authors** (`README.md:18`):
  > *"On a single H100 or a multi-node cluster, FeLLM is built to saturate the hardware."*
- **Measured Numbers**: **NOT FOUND** anywhere in FeLLM repo (*searched git log, README, scripts, and benches*). No tok/s, latency, or memory bandwidth measurements are recorded.
- **Contrast with bloomery**: bloomery records exact measured numbers on RTX 3090 (node gap 0.76 µs, barrier 2.4 µs, launch+sync 6.8 µs, llama-bench baseline 216.6 tok/s), enforces bit-identical reruns, and pins error tolerances ($10^{-5}$ to $10^{-7}$) against reference oracles.

---

## 8. Verdict: 7 Things bloomery Should Adopt & Reverse Evaluation

### 8.1 Seven Concrete Adoptions / Tests for bloomery

| # | Feature / Adoption | FeLLM Reference | bloomery Target File | Expected Effect | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Consolidated Device Parameter Struct (`DeviceStepParams`)** | `crates/fellm-plugin-abi/src/physical_plan.rs:105`, `crates/backend-cuda/src/backend.rs:151` | `crates/gpu/src/graph.rs`, `crates/gpu/src/model.rs` | Replaces scattered device scalar updates (`pos_buf`, `n_keys_buf`, `sel`) with a single 48-byte HtoD transfer per step. | **Low** (pure host/device data structure cleanup). |
| **2** | **Fused Residual Addition in GEMV Epilogue (`fuse_residual`)** | `plugins/cuda_kernels/src/oxide_kernels.rs:1480-1485`, `L1100` | `crates/gpu/src/lib.rs` (`q4k_gemv`, `q6k_gemv`), `cores.rs` | Eliminates 1 separate residual add kernel launch per layer (saving 0.76 µs node gap + dispatch overhead). | **Very Low** (adds residual slice pointer and 1 FMA/add at store). |
| **3** | **Fused Gate·Up·SwiGLU Kernel Structure** | `plugins/cuda_kernels/src/oxide_kernels.rs:1610-1687` | `crates/gpu/src/fused.rs` (P0b track) | Validates that dual-weight accumulation ($W_{\text{gate}}, W_{\text{up}}$) with activation reuse and in-register SwiGLU compiles cleanly in cuda-oxide. | **Low** (matches bloomery's P0b roadmap). |
| **4** | **Device-Side Router Top-K Selection Kernel** | `plugins/cuda_kernels/src/oxide_kernels.rs:2164-2260` (`moe_route_topk`) | `crates/gpu/src/router.rs` | Computes router GEMV, softmax, and top-k selection entirely on device in 1 launch without host interaction. | **Low** (bloomery already has `router.rs` cores; can fuse into single launch). |
| **5** | **Double-Buffered FA3 Decode Pipelining** | `plugins/cuda_kernels/src/oxide_kernels.rs:4887-5078` | `crates/gpu/src/flash.rs` | Overlaps KV global memory load latency with online softmax compute for deep contexts ($K \ge 2048$). | **Medium** (requires careful shared memory budgeting and adapting to bloomery's 32-lane reduction). |
| **6** | **32-Value Activation Quantization Option** | `plugins/cuda_kernels/src/oxide_kernels.rs:130`, `crates/fellm-runtime/src/activation.rs` | `crates/gpu/src/lib.rs` (`q3k_quantize_q8_1`), `docs/gpu-design.md` | Cuts activation quantization noise on high `amax/rms` layers from $2.4 \times 10^{-2}$ down to $\sim 1.6 \times 10^{-2}$, matching ik CUDA. | **Medium** (increases scale storage and reduces arithmetic amortization). |
| **7** | **Paged KV Cache Block Table Abstraction** | `plugins/cuda_kernels/src/oxide_kernels.rs:4980-4993`, `crates/fellm-runtime/src/kv_fabric/` | `crates/gpu/src/flash.rs`, `crates/gpu/src/model.rs` | Enables dynamic non-contiguous VRAM allocation and prefix caching for long-context serving. | **High** (requires virtual memory manager; bloomery currently relies on flat buffers). |

---

### 8.2 What FeLLM Lacks that bloomery Has (Feedback for FeLLM Authors)

1. **Rerun Bit-Identity & Determinism Discipline**: FeLLM has no bit-identity tests. bloomery gates every kernel and composite stage on exact bit-identical reproducibility across reruns.
2. **High-Performance Warp-Level GEMV Geometry**:
   - FeLLM uses 1 block per row with 128 threads and block barriers for Q4_K, and falls back to a **single-threaded scalar loop** (`q6k_gemv_row`) for Q6_K.
   - bloomery runs 8 independent rows per block (1 warp per row) with zero block barriers, achieving optimal SM occupancy on sm_86.
3. **8x Lower Float Arithmetic Overhead**: bloomery hoists scale/min conversions to 32-element sub-blocks and precomputes `s8` integer activation sums. FeLLM computes scale/min math on every 4-element chunk (~3,613 float ops vs 453 float ops for K=2048).
4. **Hardware Funnel Shifters (`funnel16`)**: bloomery seamlessly handles unaligned 210-byte Q6_K super-blocks using Ampere's 16-bit funnel shifters in registers. FeLLM does unaligned byte reads.
5. **Warp-Cooperative Indirect MoE Addressing (`_sel` Kernels)**:
   - FeLLM executes batch-1 MoE with 1 thread per row running scalar unaligned byte reads (`dot_q4k`).
   - bloomery executes all 6 experts in parallel inside `q3k_gemv_sel` and `q5_0_gemv_sel` with full warp cooperation and coalesced memory.
6. **Block-Parallel Attention Decode**: FeLLM evaluates keys serially 1-by-1 with 1 block per head (idling 80% of SMs on sm_86). bloomery evaluates 32 keys in parallel across warp lanes with butterfly reductions, achieving full SM occupancy.
7. **Empirical Measurement & Oracle Calibration**: bloomery is grounded in real microsecond hardware measurements (0.76 µs node gap, 2.4 µs barrier) and calibrated against llama.cpp. FeLLM has zero published benchmarks.

---

## 9. What Could Not Be Determined
1. **FeLLM Runtime Execution Speed on Real Hardware**: Because the environment restricts GPU execution and FeLLM contains no benchmark logs or committed run outputs, FeLLM's actual decode tok/s on an RTX 3090 could not be measured directly.
2. **Intent behind `q4k_q8_gemv_warp4`**: Whether `warp4` was abandoned due to register spilling or poor occupancy, or merely replaced during a refactor, is not documented in git history.

---

## 10. Improvement Opportunities Noticed in bloomery While Reading
1. `crates/gpu/src/router.rs:65`: In `softmax64`, `v[e] /= inv;` executes 64 separate f32 division instructions. Replacing with `let rcp = 1.0 / inv; v[e] *= rcp;` replaces 64 divisions with 1 division and 64 fast multiplications. (Trivial).
2. `crates/gpu/src/lib.rs:248-249`: `thread::index_1d().get()` is called twice consecutively (`% 256` and `/ 256`). Hoisting to `let idx = thread::index_1d().get();` avoids redundant intrinsic calls. (Trivial).
3. `crates/gpu/src/flash.rs:332-333`: `thread::index_1d().get()` is called twice consecutively; same hoisting opportunity. (Trivial).
4. `crates/gpu/src/lib.rs:428-430`: In `q4k_gemv`, when $M=1$, `y[row]` is written as an uncoalesced scalar store by lane 0 of each warp. Packing multiple warp outputs across the block into a vectorized store could improve write efficiency. (A round).
5. `crates/gpu/src/elem.rs:215`: In `rms_norm`, when $M=1$ (batch-1 decode), the grid launches 256 threads (8 warps), but warps 1..7 immediately return (`if t >= m`). Tuning launch configuration for $M=1$ to launch exactly 32 threads avoids spawning 7 idle warps. (Trivial).

---

## 11. Self-Verification & Process Disclosure

1. **Files changed**:
   - `scratchpad/fellm/report.md`: Created comparative investigation report.
2. **Completion criteria commands**:
   - Investigation only: read code from `<scratch>/peers-a/src/orielhaim__FeLLM` (commit `bdd6507`) and `bloomery/`. No repositories modified, no git state changes, no GPU execution.
3. **Self-verification**:
   - *Existing behavior removed/weakened*: None (read-only investigation).
   - *New constants/mappings/tables*: None added to any codebase.
   - *Other surfaces that should agree*: All citations cross-referenced with exact `path:line`.
   - *Changed test assertions*: None.
   - *Conflicting contract clauses*: None.
4. **What could not be implemented or verified**:
   - GPU execution of FeLLM kernels (per prompt constraints: no GPU, no building).
5. **What was deliberately left untouched**:
   - All source code in FeLLM and bloomery remained strictly untouched according to whitelist and investigation constraints.
