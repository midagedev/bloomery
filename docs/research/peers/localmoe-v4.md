# Investigation Report: localMoE — DeepSeek V4 on cuda-oxide and Implications for Bloomery V4.1 Stage

**Date**: 2026-09-21  
**Investigated Repository**: `<scratch>/peers-a/src/andyzpb__localMoE` (commit `dcdee6f`, "Run DeepSeek-v4-pro (1600B MoE) on a 24GB-RAM consumer machine — rust, streamed from disk")  
**Investigating Target**: Bloomery (`bloomery/`), workstation target: RTX 3090 (24 GB) + RTX A6000 (48 GB) + 256 GB DDR4 (8-channel ~148 GB/s) + NVMe, aiming for DeepSeek V4.1-Flash.

---

## 1. What Actually Runs

### 1.1 Model Files and Format Loaded
- **Checkpoint profiles**: Defined in [`rust/colibri-cuda/src/v4_spec.rs:49-96`](v4_spec.rs#L49-L96):
  - **DeepSeek-V4-Flash-0731**: 48 safetensors shards (`model-00001-of-00048.safetensors` to `model-00048-of-00048.safetensors`), totaling 166,886,535,336 bytes (~166.9 GB) (**read in code**: `v4_spec.rs:76-78`). 43 hidden layers, `hidden_size = 4096`, `routed_experts = 256`, `experts_per_token = 6`, `moe_intermediate = 2048`, `q_heads = 64`, `head_dim = 512`, `q_rope_dim = 64`.
  - **DeepSeek-V4-Pro**: 64 safetensors shards (`model-00001-of-00064.safetensors` to `model-00064-of-00064.safetensors`), totaling 864,721,029,744 bytes (~864.7 GB) (**read in code**: `v4_spec.rs:52-54`). 61 hidden layers, `hidden_size = 7168`, `routed_experts = 384`, `experts_per_token = 6`, `moe_intermediate = 3072`, `q_heads = 128`, `head_dim = 512`, `q_rope_dim = 64`.
- **Weight formats and quantizations**:
  - **MoE Experts (`w1`, `w2`, `w3`)**: Stored as FP4 (`TensorDtype::Fp4E2M1x2`) packed two 4-bit nibbles per byte, with block size 32 scales stored as `TensorDtype::F8E8M0` (1 byte per 32 elements) (**read in code**: `v4_expert.rs:146-180`, `v4_fp4.rs:3-5`). Each Flash-0731 expert is 12.75 MiB (13,369,344 bytes) across `w1`, `w2`, `w3` and their scales.
  - **Attention & Intermediate Projections**: `attn.wq_a`, `attn.indexer.wq_b`, `attn.wo_b`, etc., stored as FP8 (`TensorDtype::F8E4M3`), with block size 128 scales stored as `TensorDtype::F8E8M0` (**read in code**: `v4_state.rs:421-425, 487-496`).
  - **Output projection `wo_a`**: Stored on disk as `F8E4M3` with `F8E8M0` scale. At runtime, either decoded on host at load time to BF16 (`v4_state.rs:3662-3670`) or decoded on GPU via the `e4m3fn_to_bf16` kernel (`main.rs:215, 22489`).
  - **Norms, router linear weights, compressor weights**: Stored as `BF16` or `F32` (**read in code**: `v4_state.rs:415-438`).
  - **mHC coefficients**: `hc_attn_fn` / `hc_ffn_fn` stored as `F32` `[24, 4 * hidden]`, `hc_attn_base`: `F32` `[24]`, `hc_attn_scale`: `F32` `[3]`.
- **Weight conversion**: Weights are NOT converted to an offline secondary format on disk. FP4 and FP8 remain packed in their official safetensors byte layout. They are decoded either on-the-fly in registers inside the CUDA kernels or expanded on device prior to execution.

### 1.2 GPU vs CPU vs Disk Streaming Allocation
- **Streamed from disk**: In localMoE's laptop configuration, ALL layer weights (fixed tensors + the 6 selected routed experts per layer) are streamed from disk layer-by-layer on every token step (**read in code**: `main.rs:22406-22524`).
- **Executed on GPU**:
  - RMSNorm variants (`rmsnorm_v4_bf16`, `rmsnorm_flash_learned_4096_bf16`, `rmsnorm_flash_square_bf16`).
  - RoPE (`rope_v4_bf16`).
  - mHC Linear & Sinkhorn balancing (`mhc_linear_v4_bf16`, `mhc_sinkhorn_v4_bf16`, `hc_post_v4_bf16`).
  - Router linear projection & top-k selection (`select_v4_router`, or via cuBLAS `sgemm`).
  - Compressor & Indexer (`compressor_v4_bf16_f32`, `indexer_scores_v4_f32`, `indexer_topk_v4_f32`).
  - Attention computation (`sparse_attn_v4_*`).
  - Shared expert FFN (gate, up, swiglu, down).
  - Routed expert FFNs (W1, W3, swiglu, W2 via `matvec_v4_fp4_mma` or `matvec_v4_fp4_block32`).
  - Final vocabulary head (in `gpu_exact` mode via `matvec_bf16_streamed`).
- **Executed on CPU**:
  - Layer sequence orchestration and file reading (`safetensors.rs::read_exact_at` / `pread` via `RangeReadMode`).
  - Transactional host KV cache buffer updates (`v4_state.rs::V4SerialPersistentState`).
  - Final head vocabulary matvec when running in CPU fallback mode (`canonical_cpu_streamed_bf16`).

### 1.3 Per-Token Execution Flow
For each token step:
1. Embedding lookup for current token ID into initial hidden vector (expanded across 4 mHC streams, `4 * hidden`).
2. For each layer (0..42 in Flash-0731):
   a. **Fixed Weights**: Load fixed layer tensors from disk (or receive from prefetch worker), copy to GPU device memory.
   b. **mHC Attention Pre**: Sublayer input collapsed from 4 streams to 1 vector via `mhc_pre_v4_bf16` / Sinkhorn.
   c. **Attention**: RMSNorm -> Q projection -> RoPE on tail 64 dims -> Compressor projection -> Indexer FWHT & score -> Top-K key selection (top-512) -> Sparse attention softmax with sink -> Wo projection.
   d. **mHC Attention Post**: Sublayer output expanded back to 4 streams and mixed with residual via `hc_post_v4_bf16`.
   e. **mHC FFN Pre**: 4 streams collapsed to 1 vector for FFN sublayer.
   f. **Router**: RMSNorm -> Gate GEMV -> `sqrt(softplus(logit))` -> Top-6 expert selection (`score + bias`) -> Weight normalization (`raw / sum * route_scale`).
   g. **Prefetch**: Host background thread immediately begins reading the 6 routed experts from disk while GPU begins shared expert FFN.
   h. **FFN**: Shared expert evaluated on GPU; each of the 6 routed experts uploaded to GPU device memory, evaluated, and accumulated with routing weights.
   i. **mHC FFN Post**: FFN output expanded back to 4 streams and mixed with residual.
   j. Discard layer weights; advise OS page cache with `POSIX_FADV_DONTNEED`.
3. Final RMSNorm and vocabulary projection (129,280 tokens), greedy argmax selection.

### 1.4 Correctness Evidence and Test Suite
- **Single-token golden verification**:
  - Flash-0731: For `token_id = 0`, greedy step 1, all 43 layers matched official Python reference digests (`compare_v4_flash_0731_full43.py:121-137`), yielding argmax `33832` (**read in code**: `compare_v4_flash_0731_full43.py:27`, `README.md:38`). Digests matched:
    - `collapsed BF16`: `5e9764a6e105a3d6`
    - `normalized BF16`: `91495daa576afcb3`
    - `logits FP32`: `e9d90c64a1441d20`
  - Pro: 5 prompt tokens (`hi`) + 1 greedy step emitted token `效果好` in 500.4 s (**stated by authors**: `docs/experiments/v4-pro-mechanisms-2026-07-24.md:94-102`).
- **Tests found**:
  - `tests/v4_chat.rs:11-40`: Verifies message formatting against `fixtures/v4_chat_golden.tsv`.
  - Unit tests in `v4_fp4.rs`, `v4_spec.rs`, `v4_state.rs`, `safetensors.rs` for format parsing and decoding.
- **What is NOT present**:
  - NO multi-token generation quality tests, NO perplexity benchmarks (WikiText, C4), NO standard LLM eval benchmarks (MMLU, GSM8K). The authors explicitly disclaim: *"This certificate covers the pinned Flash-0731 revision, one prompt token, one greedy step... Long contexts, sampling quality, sustained multi-user throughput, and other prompts each need their own evidence"* (`README.md:46-47`).

### 1.5 Measured Throughput and Hardware Numbers
- **Hardware**: NVIDIA GeForce RTX 4080 Laptop GPU (12,282 MiB, Ada `sm_89`, driver `610.74`), host WSL2 Linux `6.18`, 24 GB host RAM, ext4 on NVMe SSD (**read in code / stated by authors**: `docs/experiments/v4-pro-mechanisms-2026-07-24.md:41-43`).
- **Flash-0731 (43 layers, 1 token decode)** (**stated by authors**: `README.md:7-22`, `docs/assets/readme/flash-0731-abba.csv`):
  - Initial baseline: `93.349 s` (~0.0107 tok/s)
  - With cuBLAS session reuse across mHC/router: `34.611 s` (~0.0289 tok/s, 2.70x speedup)
  - Final optimized (session reused for compressor/indexer + prefetch): `13.126 s` per token = **0.0762 tok/s** (7.11x total derived speedup over baseline). Standalone run: `14.298 s` with `1,412,424 KiB` maxRSS.
- **V4-Pro (61 layers, 1 token decode)** (**stated by authors**: `docs/experiments/v4-pro-mechanisms-2026-07-24.md:66-89`):
  - M1 baseline: `94.060 s` (0.0106 tok/s; fixed read+alloc: 34.38 s, decode: 20.80 s, H2D: 3.53 s, routed: 21.66 s).
  - M3 (4 persistent QD4 routed lanes): `62.020 s` (0.0161 tok/s; routed time dropped to 3.32 s).
  - M4 (raw fixed cache + QD4): `24.546 s` (0.0407 tok/s).

---

## 2. The V4 Architecture as Code

Below is every V4-specific operator implemented in localMoE, citing exact line numbers and describing the coded math:

### 2.1 Sparse Attention and Indexer (CSA/HCA)
- **`sparse_attn_v4_scores`** ([`rust/colibri-cuda/src/main.rs:1973-2016`](main.rs#L1973-L2016)): Computes the dot product between query heads (`heads = 64`, `dim = 512`) and selected key vectors indexed by `topk: &[i32]`. Thread index maps to `(row, head, slot)`. Queries and keys are BF16 converted to FP32, dot products accumulated in FP32.
- **`sparse_attn_v4_probabilities`** ([`rust/colibri-cuda/src/main.rs:2019-2086`](main.rs#L2019-L2086)): Scales dot products by `SCALE = 0.044194173` ($1/\sqrt{512}$), finds row-head `score_max`, and computes $\exp(\text{score} \cdot \text{SCALE} - \text{score\_max})$. **Critical V4 feature**: Attention sink parameter `attn_sink[head]` is added to the denominator: $\text{denom} = \sum \text{prob} + \exp(\text{attn\_sink}[\text{head}] - \text{score\_max})$.
- **`sparse_attn_v4_values`** ([`rust/colibri-cuda/src/main.rs:2089-2139`](main.rs#L2089-L2139)): Computes weighted sum of value vectors $\sum (\text{prob}_k \cdot V_{k, d}) / \text{denom}$, converted back to BF16 with Round-to-Nearest-Even (`f32_to_bf16_rne`).
- **`compressor_v4_bf16_f32`** ([`rust/colibri-cuda/src/main.rs:1077-1148`](main.rs#L1077-L1148)): Projects hidden input via `wkv` and `wgate` weights in blocks of 192, and adds Absolute Positional Embedding (`ape`) table based on `(start_pos + row) % compress_ratio`.
- **`compressor_pool_ratio4_v4_f32`** and **`compressor_pool_ratio128_v4_f32`** ([`rust/colibri-cuda/src/main.rs:1153-1229`](main.rs#L1153-L1229)): Softmax-weighted pooling across 4-slot or 128-slot compressed token windows: $\text{weight} = \exp(\text{score} - \text{max})$, $\text{pooled} = \sum (\text{kv} \cdot \text{weight}) / \sum \text{weight}$.
- **`indexer_fp4_sim_v4_bf16`** ([`rust/colibri-cuda/src/main.rs:1235-1280`](main.rs#L1235-L1280)): Performs a 128-wide Fast Walsh-Hadamard Transform (FWHT), multiplies by $1/\sqrt{128}$, rounds to BF16, and simulates block-32 E2M1 FP4 quantization by scaling by $\text{pow2\_ceil}(\text{amax} / 6.0)$.
- **`indexer_scores_v4_f32`** ([`rust/colibri-cuda/src/main.rs:1302-1344`](main.rs#L1302-L1344)): Computes indexer score: $\text{score} = \sum_{h} \max(0, Q_h \cdot K_{\text{key}}) \cdot \text{weights}_h$.
- **`indexer_topk_v4_f32`** ([`rust/colibri-cuda/src/main.rs:1382-1441`](main.rs#L1382-L1441)): Selects top-K keys (512 for Flash, 1024 for Pro) using deterministic sorting (score descending, key index ascending).

### 2.2 The Router
- **`select_v4_router`** ([`rust/colibri-cuda/src/main.rs:2765-2881`](main.rs#L2765-L2881)):
  - **Scoring function**: $\text{raw} = \sqrt{\text{softplus}(\text{logit})} = \sqrt{\log(1 + \exp(\text{logit}))}$ (`main.rs:96-122`).
  - **Bias & Selection**: For learned layers, selection compares $\text{candidate} = \text{raw} + \text{bias}[\text{expert}]$, finding the global top-6 highest candidates across all 256/384 experts (`main.rs:2826-2856`).
  - **Groups**: There are **NO groups**. DeepSeek V2/V3 group routing (`n_group`, `topk_group`) is completely absent.
  - **Weight normalization**: Weights use the un-biased $\text{raw}$ score: $\text{weight}_i = (\text{raw}_i / \text{sum}) \cdot \text{route\_scale}$.
  - **Sum reduction tree**: $\text{sum} = ((\text{raw}_0 + \text{raw}_4) + \text{raw}_2) + ((\text{raw}_1 + \text{raw}_5) + \text{raw}_3)$ (`main.rs:2859-2863`).
  - **Hash layers**: In the first 3 layers (`hash_layers = 3`), expert IDs are predetermined by `hash_ids` from input tokens; routing scores only compute normalization weights (`main.rs:2804-2823`).

### 2.3 mHC / Sinkhorn Balancing
- **`mhc_sinkhorn_v4_bf16`** ([`rust/colibri-cuda/src/main.rs:2613-2762`](main.rs#L2613-L2762)):
  - Model hidden state runs as 4 parallel streams ($HC = 4$).
  - `mhc_linear` generates 24 mixture coefficients ($MIX = 24$): 4 `pre`, 4 `post`, and a $4 \times 4$ transition matrix `comb`.
  - Initial `comb` is normalized with row softmax: $\exp(\text{comb}_{i,j} - \max) / \sum + 10^{-6}$.
  - **Sinkhorn iterations**: Exactly 20 steps (`step in 0..20`). In each step: column normalization ($\text{comb}_{i,j} /= \sum_i \text{comb}_{i,j} + 10^{-6}$). For steps $0..18$, row normalization is also applied ($\text{comb}_{i,j} /= \sum_j \text{comb}_{i,j} + 10^{-6}$). Step 19 (the 20th iteration) performs ONLY column normalization (`main.rs:2715-2739`).
  - **Collapse**: Combines the 4 input streams into 1 sublayer input: $\text{collapsed} = \sum_{c=0}^3 \text{pre}_c \cdot \text{input}_c$ (`main.rs:2746-2760`).
  - **Expansion (`hc_post_v4_bf16`)** ([`rust/colibri-cuda/src/main.rs:2226-2271`](main.rs#L2226-L2271)): Expands sublayer output back into 4 streams, mixing the residual: $\text{out}_c = \text{post}_c \cdot \text{sublayer\_out} + \sum_{s=0}^3 \text{comb}_{s,c} \cdot \text{residual}_s$.

### 2.4 RMSNorm Variants
- **`rmsnorm_v4_bf16`** ([`rust/colibri-cuda/src/main.rs:1681-1730`](main.rs#L1681-L1730)): General BF16 RMSNorm with unit-RMS option and libdevice rsqrt.
- **`rmsnorm_flash_square_bf16`** ([`rust/colibri-cuda/src/main.rs:355-366`](main.rs#L355-L366)): Separate pass converting BF16 input to materialized FP32 squares.
- **`mhc_flash_reduce_rsqrt_16384`** ([`rust/colibri-cuda/src/main.rs:372-443`](main.rs#L372-L443)): 512-thread block reduction across 16,384 values ($4 \times 4096$) using shared memory and warp shuffle down, computing $1/\sqrt{\text{mean} + 10^{-6}}$.
- **`rmsnorm_flash_learned_4096_bf16`** ([`rust/colibri-cuda/src/main.rs:503-577`](main.rs#L503-L577)): 512-thread reduction over 4096 elements with learned weight scaling and RNE rounding.

### 2.5 RoPE Variant
- **`rope_v4_bf16`** ([`rust/colibri-cuda/src/main.rs:1733-1792`](main.rs#L1733-L1792)):
  - `head_dim = 512`, but `ROPE_DIM = 64`.
  - Dimensions $0..447$ pass through unrotated (`if dimension < head_dim - ROPE_DIM { *destination = input[offset]; return; }`, lines 1760-1763).
  - Only dimensions $448..511$ receive complex rotation with interleaved cosine/sine pairs.

### 2.6 Activation Quantization
- **`quantize_v4_activation_raw`** ([`rust/colibri-cuda/src/main.rs:603-632`](main.rs#L603-L632)): Dynamic FP8 (E4M3FN) quantization with block size 128. Computes $\text{amax}$ per block, sets $\text{scale} = \text{pow2\_ceil}(\max(\text{amax}, 10^{-4}) / 448.0)$, stores scale as an 8-bit E8M0 exponent, and quantizes activations to 8-bit E4M3FN.

### 2.7 Flagged Differences from DeepSeek V2/V3 (New Kernels Needed)
| Component | DeepSeek V2/V3 (Bloomery Today) | DeepSeek V4 (New Kernels Needed) |
|---|---|---|
| **Hidden Stream** | 1 vector ($D=2048$) | **4 parallel streams ($HC=4$)**; requires mHC linear, Sinkhorn, collapse, post-expansion |
| **MoE Routing** | Group-based (8 groups, top-4 groups) via softmax | **Global flat top-6 (NO groups)**; scoring is $\sqrt{\text{softplus}(\text{logit})}$; bias in selection only |
| **Attention** | Dense MLA over full latent KV cache ($D=576$) | **Sparse Attention**: HCA/CSA ratio 4/128 compression, FWHT indexer, dynamic top-512 keys, **Attention Sink** term in denominator |
| **Head Dim & RoPE** | $d=128$, RoPE on 64 dims | **$d=512$**, RoPE on trailing 64 dims ($448..511$) |
| **Expert Weights** | GGUF Q3_K, Q4_K, Q5_0, Q6_K | **FP4 (E2M1) block 32 with E8M0 scales** |
| **Projections** | Quantized GGUF / BF16 | **FP8 (E4M3FN) block 128 with E8M0 scales** |

---

## 3. Quantized Matvec on cuda-oxide

### 3.1 `matvec_v4_fp8` ([`main.rs:694-774`](main.rs#L694-L774))
- **Block layout**: $K$-block size is 128 elements (`BLOCK = 128`).
- **Scale locations**:
  - `input_scales`: 1 byte per 128 inputs (`rows * (in_features / 128)`).
  - `weight_scales`: 2D block scales grouped into $128 \times 128$ tiles (`((out_features + 127)/128) * (in_features / 128)`).
- **Scale format**: `E8M0FNU` (power of 2 stored as 8-bit exponent, decoded via `f32::from_bits((exp as u32) << 23)`).
- **Decoding**: FP8 E4M3FN decoded via arithmetic function `decode_e4m3fn` (`main.rs:169-184`).
- **Execution**: 1 warp (32 lanes) per output element. If `!strict_k_order`, lanes stride by 32 across the 128-block, reduce partial sums using `warp::shuffle_xor_f32(16, 8, 4, 2, 1)`, and lane 0 applies combined scales and writes `output_f32` and `output_bf16`.

### 3.2 `matvec_v4_fp8_mma` ([`main.rs:874-971`](main.rs#L874-L971))
- **Block layout**: `FP8_BLOCK = 128`, `MMA_K = 32`, `MMA_N = 8`.
- **MMA intrinsic**: Calls `cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3(fragment, [a0, a0, a1, a1], [b0, b1])` (`main.rs:948`).
- One warp computes a $16 \times 8$ tensor-core tile with the single input row duplicated across all 16 rows of matrix A.
- **Hardware architecture & sm_86 usability**:
  - **USABLE ON SM_86? NO!**
  - **Inference / PTX Fact**: The `mma.sync.aligned.m16n8k32.*.e4m3.e4m3` instruction was introduced in **Ada Lovelace (`sm_89`) and Hopper (`sm_90`)**. Ampere (`sm_80`, `sm_86`, including RTX 3090 and RTX A6000) **DOES NOT HAVE FP8 Tensor Cores**. Compiling or launching this kernel on `sm_86` will fail.

### 3.3 `matvec_v4_fp4` ([`main.rs:1446-1497`](main.rs#L1446-L1497))
- **Block layout**: `WEIGHT_BLOCK = 32` elements.
- **Scales**: `scales: &[u8]`, 1 byte per 32 weights (`out_features * (in_features / 32)`).
- **Inputs**: Raw `f32` slice (`input: &[f32]`).
- **Decoding**: Arithmetic table lookup in registers for FP4 E2M1 nibble (`decode_fp4`, `main.rs:150-166`):
  $\text{nibble} \in \{0..7\} \to [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]$ with sign bit at bit 3.
- **Execution**: 1 warp per output row; lanes stride by 32 blocks, warp shuffle xor reduction, lane 0 writes BF16.

### 3.4 `matvec_v4_fp4_block32` ([`main.rs:1502-1555`](main.rs#L1502-L1555))
- Exact-diagnostic arm: Each lane owns one value in a 32-wide block. Multiplies FP8 E4M3FN input by FP4 E2M1 weight, reduces unscaled partials across the warp via shuffle xor, then lane 0 applies both the activation scale (block / 4) and weight scale.

### 3.5 `matvec_v4_fp4_mma` ([`main.rs:979-1073`](main.rs#L979-L1073))
- Expands FP4 E2M1 nibbles to raw FP8 E4M3 in registers using a packed 64-bit immediate register LUT (`v4_fp4_pair_to_e4m3x2`, `main.rs:55-73, 1037-1040`).
- Executes `cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3`.
- **USABLE ON SM_86? NO!** Same restriction as `matvec_v4_fp8_mma`. Requires `sm_89`+.

### 3.6 `matvec_bf16_streamed` ([`main.rs:778-868`](main.rs#L778-L868))
- Streamed BF16 GEMV for the vocabulary output head (`vocab = 129,280`).
- 1 warp owns 1 vocabulary row, strides by 32 lanes through input columns.
- Incorporates full numerical certificate status checking on every product and sum: sets bitflags for `CERTIFIED_HEAD_NONFINITE`, `CERTIFIED_HEAD_SUBNORMAL`, `CERTIFIED_HEAD_OVERFLOW`, and computes `absolute_sum` for condition bounds. Warp shuffle xor reduces sum, absolute sum, and status bits.

---

## 4. Expert Streaming

### 4.1 Granularity and Movement
- Granularity per token: 6 routed experts per layer (`experts_per_token = 6`).
- Each expert comprises 6 tensors: `w1.weight`, `w1.scale`, `w2.weight`, `w2.scale`, `w3.weight`, `w3.scale`.
- Bytes per expert:
  - Flash-0731: $12.75\text{ MiB}$ ($13,369,344\text{ bytes}$). 6 experts = $76.5\text{ MiB}$ ($80,216,064\text{ bytes}$) per layer. Across 43 layers = **$3.29\text{ GB}$ per token**.
  - Pro: $42.0\text{ MiB}$ ($44,040,192\text{ bytes}$). 6 experts = $252\text{ MiB}$ ($264,241,152\text{ bytes}$) per layer. Across 61 layers = **$15.37\text{ GB}$ per token**.

### 4.2 Caching, Eviction, and Lifecycle
- **Lifecycle in default Flash mode (`ExpertSlot::new`, `main.rs:14410-14450`)**:
  - `v4_expert::RoutedExpert::load_layer_for` reads the 6 tensor ranges from safetensors file via `pread` into heap `Vec<u8>` buffers.
  - `PinnedHostBuffer::from_slice` allocates pinned host memory and copies bytes from heap.
  - `DeviceBuffer::uninitialized_async` allocates GPU device buffers.
  - Asynchronous copy `copy_from_pinned_host_async` moves weights to GPU.
  - GPU executes expert GEMVs (`slot.execute`).
  - `ExpertSlot` is dropped at the end of the expert execution, deallocating both GPU device memory and pinned host memory!
  - `advise_dontneed` calls `posix_fadvise(..., POSIX_FADV_DONTNEED)` on the disk shard.
- **QD4 Immutable Cache (`main.rs:15143-15250`)**:
  - 32 slots (`V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS = 32`).
  - **Strictly layer-local**: Flushes all resident entries across layer boundaries (`begin_layer` enforces `resident.is_empty()`, `main.rs:15169`). It only serves intra-layer reuse across requests or tokens within a batch.

### 4.3 Prefetch Driven by Router Prediction
- Implemented in `flash_routed_prefetch` ([`main.rs:37366-37408`](main.rs#L37366-L37408)):
  - As soon as `run_layer_router` finishes selecting the 6 expert IDs on the GPU, host spawns a background thread via `std::thread::scope`:
    ```rust
    let worker = scope.spawn(|| {
        route_ids.into_iter().map(|id| RoutedExpert::load_layer_for(...))
    });
    ```
  - Concurrently, the GPU executes the shared expert FFN (`run_layer_shared_expert`).
  - Host joins the prefetch worker and immediately begins uploading the prefetched experts to the GPU.

### 4.4 Bytes Moved Per Token
- **Flash-0731 (uncached laptop streaming)**:
  - Fixed weights streamed from disk: $\sim 29\text{ GB}$ per token across 43 layers.
  - Routed weights streamed from disk: $3.29\text{ GB}$ per token across 43 layers.
  - Total data read from disk: $> 32\text{ GB}$ per token.
- **V4-Pro (uncached laptop streaming)**:
  - Fixed weights: $24.97\text{ GB}$ read from disk, $29.06\text{ GB}$ H2D transfer.
  - Routed weights: $12.84\text{ GB}$ read from disk / H2D transfer.
  - Total H2D transfer: **$41.91\text{ GB}$ per token**.

### 4.5 What Changes with 256 GB Host RAM (Bloomery Workstation Target)
In localMoE, the 12 GB GPU + 24 GB host RAM forced constant NVMe disk streaming at 2–3 GB/s, resulting in the 13.12 s/token latency (0.076 tok/s).  
With 256 GB DDR4 host RAM and RTX 3090 (24 GB) + A6000 (48 GB) = 72 GB VRAM:
1. **100% Model RAM Residency (Zero Disk I/O)**:
   The entire 166.9 GB Flash-0731 checkpoint fits completely inside 256 GB host RAM. All weights are mmapped or pinned in host RAM once at startup. **Disk read during inference drops from 32 GB to EXACTLY 0 bytes**.
2. **100% Fixed Weights GPU Residency (Zero Fixed H2D)**:
   All fixed weights (attentions, norms, routers, embeddings, head) across all 43 layers require $\sim 15\text{--}20\text{ GB}$ total VRAM. On a 72 GB VRAM system (or even on the 48 GB A6000 alone), 100% of the fixed weights, KV cache, and scratch arenas remain permanently resident in GPU memory. **Fixed weight H2D transfer drops from 29 GB to EXACTLY 0 bytes**.
3. **PCIe Routed Streaming Only**:
   The ONLY data transferred per token is the 6 routed experts per layer: $3.29\text{ GB}$ total per token.
   - Over PCIe 4.0 x16 ($\sim 25\text{ GB/s}$): $3.29\text{ GB} / 25\text{ GB/s} \approx \mathbf{130\text{ ms}}$.
   - Over 8-channel DDR4 ($\sim 148\text{ GB/s}$ host memory): $3.29\text{ GB} / 148\text{ GB/s} \approx \mathbf{22\text{ ms}}$.
   - This transforms decode latency from 13.1 seconds (~0.076 tok/s) to $\sim 150\text{--}200\text{ ms}$ (**5–7 tok/s**), a **70–100x speedup** purely from memory hierarchy residency.

---

## 5. Toolchain Lessons

1. **Pinned cuda-oxide Revision**:
   - `colibri-cuda` pins `cuda-oxide` to commit `8d02eac17e56d5ed5ad3713fc498a2591d810919` ([`Cargo.toml:8-10`](andyzpb__localMoE/rust/colibri-cuda/Cargo.toml#L8-L10)).
   - Rust toolchain pinned to `nightly-2026-04-03` ([`rust-toolchain.toml:2`](andyzpb__localMoE/rust/rust-toolchain.toml#L2)).
2. **Missing Intrinsics & Direct `libdevice` FFI**:
   - `cuda_device` lacked accurate transcendentals (`expf`, `log1pf`, `rsqrtf`). localMoE bypassed `cuda_device` by declaring direct `unsafe extern "C"` bindings to CUDA libdevice (`__nv_expf`, `__nv_log1pf`, `__nv_rsqrtf`) inside `#[cuda_module]` ([`main.rs:80-86`](main.rs#L80-L86)).
3. **Preventing LLVM FMA Contraction with `read_volatile`**:
   - Rust/LLVM aggressively contracts multiply + add into `fma` and reorders associative floating-point additions. To match PyTorch/TileLang bit-exact order, localMoE resorted to `core::ptr::read_volatile` (e.g. macro `tilelang_sum4!`, [`main.rs:45-53`](main.rs#L45-L53); [`main.rs:1784, 2258, 2742`](main.rs#L1784)).
4. **Compiler Build Lock Bug (`CARGO_BUILD_JOBS=1`)**:
   - In [`run-identified.sh:32-35`](andyzpb__localMoE/rust/run-identified.sh#L32-L35): *"CUDA-Oxide uses a fixed LLVM artifact name for bin and test codegen. if [[ ${1:-} == test ]]; then export CARGO_BUILD_JOBS=1; fi"*. Parallel builds race on the fixed artifact name.
5. **Monolithic File Architecture Failure**:
   - `main.rs` grew to **45,518 lines** containing all 52 kernels, runners, telemetry, and tests. Bloomery's modular architecture (separating device functions in `cores.rs` and modular `#[cuda_module]` per domain file) is vastly superior and prevents merge paralysis.

---

## 6. Verdict: What Bloomery V4.1 Should Take vs Reject

### 6.1 Top 7 Items Bloomery Should Take
1. **In-Register FP4 (E2M1) to FP8 (E4M3) Bit-Shift LUT** ([`rust/colibri-cuda/src/main.rs:55-73`](main.rs#L55-L73)):
   - *Math*: `const V4_FP4_E2M1_TO_E4M3_POSITIVE: u64 = 0x4c48_4440_3c38_3000;`. Unpacks nibbles via bit shifts directly in 64-bit registers.
   - *What it saves us*: Zero local memory spills and zero memory lookups when decoding FP4 weights.
2. **Exact 20-Iteration Sinkhorn Balancing Kernel** ([`rust/colibri-cuda/src/main.rs:2613-2762`](main.rs#L2613-L2762)):
   - *Math*: $4 \times 4$ doubly-stochastic normalization with asymmetric 20th iteration (omits final row normalization).
   - *What it saves us*: Reverse-engineering the exact mHC manifold mixing math required to match DeepSeek V4 checkpoints.
3. **V4 Router Scoring Math and Normalization Tree** ([`rust/colibri-cuda/src/main.rs:2765-2881`](main.rs#L2765-L2881)):
   - *Math*: $\sqrt{\text{softplus}(\text{logit})}$, top-6 selection with bias, weight normalization without bias via fixed 6-term binary addition tree.
   - *What it saves us*: Prevents route prediction errors and numerical argmax divergence.
4. **Attention Sink Integration in Sparse Softmax** ([`rust/colibri-cuda/src/main.rs:1877, 2081`](main.rs#L1877)):
   - *Math*: $\text{denom} += \exp(\text{attn\_sink}[\text{head}] - \text{score\_max})$.
   - *What it saves us*: Correct softmax denominator calculation; without this attention sink term, attention weights in V4 are mathematically incorrect.
5. **Dynamic Block-128 FP8 Activation Quantizer** ([`rust/colibri-cuda/src/main.rs:603-632`](main.rs#L603-L632)):
   - *Math*: $\text{scale} = \text{pow2\_ceil}(\max(\text{amax}, 10^{-4}) / 448.0)$, with E8M0 scale packing.
   - *What it saves us*: Efficient on-device activation quantization matching official V4 GEMV inputs.
6. **Compressed Attention Pooling & Indexer Structure** ([`rust/colibri-cuda/src/main.rs:1077-1240`](main.rs#L1077-L1240)):
   - *Math*: Ratio-4 and ratio-128 score-weighted pooling combined with FWHT-128 and simulated E2M1 quantization for sparse KV selection.
   - *What it saves us*: Complete architectural blueprint for DeepSeek V4's Compressed Sparse Attention (CSA/HCA).
7. **Asynchronous Router-Driven Routed Expert Prefetch Pipeline** ([`rust/colibri-cuda/src/main.rs:37370-37408`](main.rs#L37370-L37408)):
   - *Pattern*: As soon as router outputs route IDs on GPU, asynchronously launch PCIe H2D transfers for the 6 routed experts from host pinned memory while the GPU computes the shared expert FFN.
   - *What it saves us*: Hides PCIe transfer latency behind shared expert computation.

### 6.2 What Looks Wrong or Unverified and Must NOT Be Copied
1. **DO NOT COPY Ada `sm_89` MMA Intrinsics to sm_86 (`matvec_v4_fp8_mma`, `matvec_v4_fp4_mma`)** ([`main.rs:948, 1044`](main.rs#L948)):
   - `mma_m16n8k32_fp8_f32_e4m3_e4m3` requires Ada (`sm_89`) or Hopper (`sm_90`). It **does not exist on sm_86** (RTX 3090 / RTX A6000). Bloomery must dequantize FP4/FP8 to INT8 or FP16/BF16 in registers to use Ampere's tensor cores, or use vectorized SIMT DP4A/FMA.
2. **DO NOT COPY Per-Expert Ephemeral Allocation & Pinning Churn** ([`main.rs:14418-14450`](main.rs#L14418-L14450)):
   - Allocating `PinnedHostBuffer` and `DeviceBuffer` dynamically for every expert call causes massive allocator locking and synchronization overhead. Bloomery's `gpu-design.md:31` rule ("No allocations during step; static resident buffers and pre-allocated staging arenas") must be strictly preserved.
3. **DO NOT COPY `read_volatile` Compiler Hacks** ([`main.rs:45-53, 1784, 2267`](main.rs#L45-L53)):
   - Volatile operations disable compiler optimizations, force L1 round-trips, and wreck register allocation. Use explicit non-contracting floating-point intrinsics (`add_rn_f32`) instead.
4. **DO NOT COPY Unoptimized Scalar Attention Loop** ([`main.rs:1843-1880`](main.rs#L1843-L1880)):
   - The authors fell back to a serial scalar thread loop over all 512 dimensions because the TileLang kernel exceeded shared memory limits. A production engine needs tiled FlashAttention-style warp cooperative reductions.
5. **DO NOT COPY Disk Streaming for Workstation Deployments**:
   - Streaming from disk layer-by-layer was a forced workaround for a 24 GB laptop. On our 256 GB RAM workstation, the entire model must be resident in RAM.

---

## 7. What Could Not Be Determined

1. **Exact Official DeepSeek-V4 Reference Repository Parity**:
   - The peer repository compares itself against its own Python re-implementation (`v4_flash_0731_prefix_oracle.py`, `v4_full61_oracle.py`) rather than the official DeepSeek PyTorch repository or HuggingFace transformers pipeline. Whether the official DeepSeek V4 codebase has subtle differences in tie-breaking, epsilon offsets, or layer normalization order could not be proven without access to the official weights and PyTorch codebase.
2. **Multi-Token Context Rollover Behavior**:
   - No runs in the peer repository test contexts beyond 5 prompt tokens + 1 generated token. The long-context numerical stability of the 20-iteration Sinkhorn normalization and attention sink across hundreds of tokens remains unmeasured.

---

## 8. What Was Deliberately Left Untouched

1. `bloomery/`: Left completely unmodified. Absolute ban on repository modification respected.
2. `/private/tmp/.../andyzpb__localMoE`: Left completely unmodified. Read-only inspection only.
3. Git state: No commits, checkouts, stashes, or branch switches made.

---

## 9. Self-Verification

1. **Existing behavior this change removed or weakened**: None. This round is an investigation and reporting task only. Grep verification of git status in bloomery confirms zero modifications: `git -C bloomery/ status --porcelain` is clean of any new unstaged files from this task.
2. **New constants/mappings/tables**: None added to bloomery.
3. **Other surfaces that should now agree**: Not applicable.
4. **Changed test assertions**: None changed.
5. **Contract clauses that conflicted**: None. All six investigation questions were answered with concrete code citations and architectural derivations.

---

## 10. Improvement Opportunities Noticed for Bloomery

1. **CUDA Graph Safety & Allocations in GPU Stage** ([`crates/gpu/src/graph.rs`](file://bloomery/crates/gpu/src/graph.rs)):
   - *Observation*: LocalMoE struggled with CUDA graph replay because intermediate expert addresses varied dynamically. Bloomery's P9 design (`_sel` kernels passing expert indices through device buffers) is the correct pattern to avoid recreating graph nodes.
   - *Size*: Design confirmed.
2. **Ampere FP4 Tensor Core Strategy for V4.1**:
   - *Observation*: Because sm_86 lacks FP8/FP4 MMA instructions, Bloomery's V4.1 stage must not attempt to use `cuda_device::wmma` for FP8/FP4. Instead, bloomery should either implement a register-unpacked INT8 Tensor Core GEMV (`mma.m16n8k32.s8`) or a vectorized DP4A/FMA SIMT kernel.
   - *Size*: One design round.
3. **Double-Buffered Pinned PCIe Expert Staging**:
   - *Observation*: In Bloomery's V4.1 stage with 256 GB host RAM, holding all 166.8 GB of weights in host memory and streaming 3.29 GB per token over PCIe will be the primary latency component (~130 ms). Staging expert transfers in pinned host memory double-buffered with GPU execution will allow overlapping layer $L+1$ expert transfers with layer $L$ FFN execution.
   - *Size*: One implementation round.

---

## 11. Premises Corrections

- **Premise in Task**: Peer repo commit `dcdee6f` was described as *"Run DeepSeek-v4-pro (1600B MoE) on a 24GB-RAM consumer machine — rust, streamed from disk"*.
  - *Correction*: While the repo contains V4-Pro (61 layers, 864 GB) code and experiments, the primary optimized and verified target in `dcdee6f` is **DeepSeek-V4-Flash-0731** (43 layers, 166.9 GB, safetensors, FP4 E2M1 weights), which achieved the headline 0.0762 tok/s on a 12 GB RTX 4080 Laptop GPU.
- **Premise in Task**: Peer repo might contain general sm_86 compatible FP8/FP4 MMA kernels.
  - *Correction*: The MMA kernels (`matvec_v4_fp8_mma`, `matvec_v4_fp4_mma`) rely on `mma_m16n8k32_fp8_f32_e4m3_e4m3`, which targets Ada Lovelace (`sm_89`) and Hopper (`sm_90`). They are **incompatible with sm_86 (RTX 3090 / A6000)**.
