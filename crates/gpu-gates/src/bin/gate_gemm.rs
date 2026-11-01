//! GPU gate for the grouped int8 tensor-core GEMM (`bloomery_gpu::gemm`):
//! K-quant expert stacks times q8_1 activations for a batch of routed slots,
//! the route table built on the card from the ids.
//!
//! Cases: the real Qwen3-30B-A3B stacks (the model file of the qwen3moe
//! profile) — a routed gate stack (Q4_K, 2048 → 768, 128 experts, top-8,
//! every slot reading its token's column), a Q4_K and a Q6_K down stack
//! (768 → 2048, each slot its own column; 630-byte Q6_K rows, every other
//! one at 2 mod 4) — and one routed gate stack of DeepSeek-V4.1-Flash (Q3_K,
//! 5120 → 2304, 384 experts, top-6; `BLOOMERY_V41_MODEL`); then synthetic
//! stacks from a fixed seed at the same shapes for Q4_K, Q6_K and Q3_K, a
//! Q5_K stack at the Qwen3 gate shape, and a Q4_K stack whose rows per expert
//! (400) end in a partial 128-row slab. Each case runs T ∈ {1, 15, 16, 17,
//! 64, 511, 512, 4096} tokens (4096 the largest ubatch) under four routings:
//! uniform, every token on the same top-k experts, a quarter of the experts
//! only, and slots dealt to experts round-robin (one slot per expert while
//! there are fewer slots than experts).
//!
//! The Qwen3.8 i-quant cases (`q38_*`, the card route `wide38.rs` wires):
//! synthetic IQ3_XXS and IQ4_XS stacks of the routed gate shape (64 experts
//! of 640 rows, K 2560, top-10, every slot its token's column; the blocks
//! cycled from the ref-synth dump) and an IQ3_XXS stack of 400 rows an
//! expert, each under the four routings — at T ∈ {1, 15, 16, 17, 64, 511,
//! 512, 4096} for the top-10 cases, the card route's run of a whole ubatch
//! at the route table's 40,960-slot cap — with the same checks, the T = 1 outputs against
//! the i-quant row cores (`iq3_xxs_rows`, `iq4_xs_rows`) over the repacked
//! stack as a diagnostic, a NaN `d` super-block's outputs NaN, and
//! `from_ggml`'s acceptance of the two types.
//!
//! The Bolt IQ2_XXS cases (`iq2xxs_*`): a synthetic stack of a routed gate
//! shape (64 experts of 640 rows, K 2560, top 9, every slot its token's
//! column) and a 400-row ragged slab at K 1280 (odd super-blocks a row: each
//! row padded to whole words, every other super-block at 2 mod 4), each under
//! the four routings — with the same checks and the T = 1 outputs against the
//! `iq2_xxs_rows` row core as a diagnostic, a NaN `d` super-block's outputs
//! NaN, `from_ggml`'s acceptance of IQ2_XXS, and the host API's refusal of
//! IQ3_XXS on the stack.
//!
//! The IQ3_S cases (`iq3s_*`, the stacks of Qwen3.6's UD-IQ4_XS file): a
//! synthetic gate stack of the routed shape (256 experts of 512 rows, K 2048,
//! top 8, every slot its token's column), a down stack (256 experts of 2048
//! rows, K 512, each slot its own column) and a 400-row ragged slab at K 1280
//! (odd super-blocks a row: each row padded to whole words, every other
//! super-block at 2 mod 4), each under the four routings — with the same
//! checks, a NaN `d` super-block's outputs NaN, `from_ggml`'s acceptance of
//! IQ3_S beside its refusal of Q2_K, and the host API's refusal of IQ3_XXS
//! and IQ4_XS on the stack by its row size.
//!
//! The IQ2_XS, IQ2_S, IQ1_S and IQ1_M cases (`iq2xs_*`, `iq2s_*`, `iq1s_*`,
//! `iq1m_*`): per type a synthetic gate stack (64 experts of 640 rows, K 2560,
//! top 9, every slot its token's column), a down stack (64 experts of 2048
//! rows, K 512, each slot its own column) and a 400-row ragged slab at K 1280,
//! each under the four routings — with the same checks, the decode held to
//! ggml's `dequantize_row_*` (the gate's port for IQ2_S, IQ1_S and IQ1_M), a
//! NaN `d` super-block's outputs NaN, `from_ggml`'s acceptance of the type and
//! the host API's refusal of the IQ3_XXS row size on the gate and down stacks.
//!
//! Checks per run:
//! 1. Every output of every slot written (`y` starts at a sentinel).
//! 2. Every checked output within its derived band of the f64 reference:
//!    the dot of the exactly decoded weights with the q8_1-reconstructed
//!    activation, summed per 128-value block as the kernel groups it. The
//!    kernel's per-block integers are exact; each block adds at most four
//!    f32 roundings (the i32 → f32 conversion, `d·isum`, the min fma, the
//!    accumulating fma) and the accumulator sees `nb` blocks, so
//!    `|y − ref| <= γ(nb + 4) · Σ_b (|D_b| + |M_b|)` with `D_b =
//!    d8·d·isum`, `M_b = d8·dmin·imin` (`M_b = 0` for Q6_K and Q3_K). The
//!    decode is checked against ggml's own `dequant_row` on every decoded
//!    row.
//! 3. Every checked output bit for bit the host transcription of the
//!    module's numeric contract: per 128-value block the exact i32 `isum`
//!    and `imin`, then `acc = fma(d8, fma(−dmin, f32(imin), d·f32(isum)),
//!    acc)` (Q6_K, Q3_K: `acc = fma(d8, d·f32(isum), acc)`) from `acc = 0`,
//!    blocks in increasing k. The band of 2 admits any order or rounding
//!    inside it; this check admits the contract's alone.
//! 4. Diagnostic, not a pin: the max relative difference to today's `_sel`
//!    gemv over the same q8_1 codes (`q4k_gemv_sel`, `q6k_gemv_sel`,
//!    `q3k_gemv_sel`; Q5_K has none), and an FNV-1a digest of every output
//!    of the run (`y_fnv`), which two builds' logs compare line by line.
//!
//! Checked outputs: every row of every slot up to 136 slots; above, every
//! slot's rows `r` with `(r + s) % 4 == 0`, so each row is checked on a
//! quarter of its expert's slots.
//!
//! Every reference worker must finish: a panic is a gate error, even after
//! other workers have checked outputs. It is not a numeric or untouched-slot
//! mismatch, and cannot be replaced with a partial reference result.
//!
//! `--case <text>` runs only the cases whose name contains `text` (the
//! dense case is `dense`, the faults and refusals `fault`, the route table
//! `route_table`).
//!
//! `--bench-kernels` runs no case: it prices the GEMM's launches instead
//! (`bench` below, lead-only through `tools/ref/time-gate.sh`).
//!
//! The route table (`GemmKernels::enqueue_route`, case `route_table`) for T
//! ∈ {1, 17, 512, 1000, 4095, 4096} tokens under the four routings, on a
//! Qwen3 stack's shape (128 experts, top-8: up to 32,768 slots), a Qwen3.6
//! one's (257 experts — the shared expert joined as the last — and nine
//! slots a token: up to 36,864 slots in 31 chunks), a Qwen3.8 card route's
//! (512 experts, ten slots a token: up to 40,960 slots, the route's cap, in
//! 16 chunks) and a V4.1 one's (384 experts, top-6), and the dense table (`enqueue_route_dense`)
//! at the same counts: the slot list and the tiles bit for bit the host's
//! stable grouping — experts ascending, each expert's slots ascending, runs
//! cut into tiles of at most `GEMM_BN`. Then at 32,768 slots: three ids past
//! the stack in three different chunks end as the named fault with the
//! table the host builds without them and their slots listed as refused, in
//! slot order, and a captured route's replay writes the eager table.
//!
//! The SwiGLU quantizer between gate·up and down
//! (`GemmKernels::enqueue_swiglu_quant`, case `swiglu`) at K ∈ {768, 2048}
//! for 1, 8, 64 and 4096 columns: its five planes bit for bit what
//! `ElemKernels::enqueue_swiglu` then `Gpu::enqueue_quantize_gemm` write;
//! the scales and code sums the host transcription of the quantizer gives
//! from those SwiGLU rows; every SwiGLU value within four f32 ulps (and
//! 2^-126) of the f64 `g / (1 + e^-g) · u`. A NaN in a gate row ends as the
//! named `QuantColumn` fault with its 128-value block refused: a NaN scale,
//! and every other byte of the five planes what the same input with that
//! block zeroed leaves; its refusals (no columns, more than the activation
//! holds, a short input). Case `swiglu_clamp`: the same launch under ik's
//! clamped SwiGLU (`GemmKernels::enqueue_swiglu_act_quant`, limits 10 and 0.5)
//! bit for bit the CPU tier's ik-verified `qdot::swiglu_clamp` then
//! `Gpu::enqueue_quantize_gemm`, a `Q8Act` fill (`enqueue_swiglu_act_quant_q8`)
//! holding the `GemmAct`'s codes and scales, a NaN up value refused as
//! `QuantColumn` where the clamp alone would make it the limit, a non-finite
//! limit refused by name. Case `swiglu_oai`: the same launch under the OAI
//! SwiGLU (limits 7 and 0.5) bit for bit the CPU twin `qdot::swiglu_oai` then
//! the same quantizer, a NaN up value refused as `QuantColumn`, and a NaN or
//! negative limit and an `alpha` other than the references' refused by name.
//! Case `swiglu_sel`: `GemmKernels::enqueue_swiglu_act_quant_sel` over a
//! places row under the clamp 10 and the plain rule — the card columns the
//! plain rule-taking launch's bits over other rows, the host columns a first
//! plain launch's, untouched, a NaN in a host column raising nothing, a
//! place in `[n_card, HOST)` `ExpertId` with nothing written.
//!
//! The squared-ReLU quantizer fold (`Gpu::enqueue_quantize_gemm_relu_sqr`,
//! case `relu_sqr`) at K ∈ {768, 2048} for 1, 8, 64 and 4096 columns: its
//! five planes bit for bit what `ElemKernels::enqueue_relu_sqr` then
//! `Gpu::enqueue_quantize_gemm` write; the scales and code sums the host
//! transcription of the quantizer gives from the transformed rows (the
//! kernel applies a finite value `max(v, 0)²` in its load); the elem output
//! each transformed value's bits. A NaN in a row ends as the named
//! `QuantColumn` fault with its 128-value block refused, the other bytes
//! what the same input with that block zeroed leaves.
//!
//! And once per case: a rerun bit-identical, the route and the GEMM
//! captured into a graph whose replay equals the eager launch. Then the
//! faults: a NaN activation ends as the named error `GpuError::Fault`, its
//! block refused as in the SwiGLU case, every slot reading its column NaN
//! and every other slot the zeroed block's run bit for bit; ids past the
//! stack end as the same error with their slots listed as refused at the
//! table's end and every output row of theirs NaN; the same fault for the
//! three `_sel` gemvs, where an id past the stack raises
//! `FaultSite::ExpertId` and a host-served slot (`hybrid::HOST`) raises
//! nothing; and the host API's refusals (type, K, slot count, row count,
//! unfilled table, token shape, activation capacity, a Q4_K stack that does
//! not start 16-byte aligned). Last, the fault's site
//! mask: two sites raised in one layer after a third in a later layer are
//! all the first layer's mask holds, on the card and in the argmax's copy.
//!
//! The 32-value family (`Gemm32Kernels`, cases `g32_*`), on synthetic stacks
//! whose decode is checked against ggml's `dequant_row`:
//! - `g32_quant`: `quantize_gemm32`'s three planes bit for bit the host
//!   quantizer (per 32 values `d = amax/127`, codes, their sum; the padding
//!   untouched) at K ∈ {96, 320, 352, 640, 2560, 6144, 10240} for 1, 9 and
//!   4096 columns; a NaN and an infinity each refuse their block (NaN
//!   scale, zero codes and sum) and raise `QuantColumn`; `swiglu_quant32`
//!   bit for bit `ElemKernels::enqueue_swiglu` then `quantize_gemm32`, its
//!   NaN refused the same way; `swiglu_quant32_sel` over a places row — the
//!   card columns the plain launch's bits, the host columns a first plain
//!   launch's, untouched, a NaN in a host column raising nothing and a
//!   place in `[n_card, HOST)` `ExpertId` with nothing written.
//! - `g32_swiglu_act`: `swiglu_act_quant32` under ik's clamped SwiGLU at
//!   limits 10 (GLM-5.3's) and 0.5, K ∈ {96, 2048, 12288} for 1, 9 and 512
//!   columns, bit for bit the CPU tier's ik-verified `qdot::swiglu_clamp`
//!   then the host quantizer, each side of the clamp met (silu(g) past the
//!   limit, u past either side, neither); under the plain rule
//!   `swiglu_quant32`'s bytes; a NaN or infinite `g` or `u` refusing its
//!   block and raising `QuantColumn` where the clamp alone would make it
//!   finite; the OAI SwiGLU (limits 7 and 0.5, K = 2048, 9 columns) bit for
//!   bit the CPU twin `qdot::swiglu_oai` then the host quantizer.
//! - `g32_relu_sqr`: `relu_sqr_quant32`, a relu² expert's up rows quantized
//!   as its down GEMM's input, at K ∈ {96, 1856, 3712} (Lightning's routed
//!   and shared down widths, neither a multiple of 256) for 1, 9 and 512
//!   columns: the three planes bit for bit the host quantizer of the rule
//!   (`max(v, 0)²` of the f32 maximum, a non-finite value as it is) and
//!   `ElemKernels::enqueue_relu_sqr` then `quantize_gemm32`, with negative
//!   and positive inputs both met; a NaN, either infinity and a value whose
//!   square overflows each refuse their block (NaN scale, zero codes and sum)
//!   and raise `QuantColumn`, where a `max` before the quantizer would have
//!   made the NaN and the negative infinity a zero.
//! - `g32_dense`: at every K above and m ∈ {1, 8, 9, 512, 4096}, one
//!   quantization and the three entries over the dense table, 272 rows (two
//!   full slabs and one of 16): every checked output inside its band of the f64
//!   reference (per block the exact integer dot, at most three f32 roundings
//!   and the accumulation, `|y − ref| <= γ(K/32 + 3) · Σ_b (|D_b| + |M_b|)`)
//!   and bit for bit the contract's transcription (`gemm32.rs`); the file's
//!   Q8_0 words and the planes made from the same bytes give the same bits;
//!   a captured table and GEMM replay the eager run.
//! - `g32_routed`: routed stacks under uniform and same-expert routings — a
//!   Q8_0 file and a Q5_1 stack of 16 × 272 rows at K 640, each slot its own
//!   column; a Q8_0 plane stack of 8 experts at K 2560 whose two slots a
//!   token share its column; a Q5_1 stack of 8 × 2560 rows at K 640 (the
//!   routed down's shape) up to 4096 slots.
//! - `g32_slab`: both Q8_0-plane entries on one stack past the raster's
//!   bytes, 4112 rows (32 full slabs and one of 16) at K 2560 and at K 2592
//!   (81 blocks a row, so rows start inside a group of eight scales), through
//!   the dense table — 16 columns (one tile, 33 blocks: `gemm_q8_0p`), 65 (two
//!   tiles, 66 blocks, under one wave of the card: `gemm_q8_0p`) and 1100
//!   (18 tiles, 594 blocks: `gemm_q8_0p_slab`, one full raster group of 16
//!   tiles and one of 2): the entry `q8_0p_slab` names for each. Numerical checks
//!   cover all outputs at 16/65 columns and a rotating quarter at 1100;
//!   checked outputs match the band and the contract's transcription bit for bit.
//! - `g32_nvfp4_dense`: the resident NVFP4 planes (`mxfp4`'s `repack`, the
//!   stack the dense gemv reads) in `gemm_nvfp4p` at Bolt's two NVFP4
//!   projections' shapes — `ssm_in` 18560 × 4096, `ssm_out` 4096 × 8192 —
//!   and 128 rows at both K, over the dense table for 1, 9, 64 and 448
//!   columns: the decode ggml's, every checked output inside its band
//!   `γ(2K/32 + 4) · Σ_b |scale·d_a·(d0·i0 + d1·i1)|` of the f64 reference
//!   and bit for bit the contract (`gemm32::dot32_nv`); a rerun
//!   bit-identical; at K 64 outputs whose F32 walk overflows before the
//!   tensor scale (activation codes 127, scale 2^-120) stored as the finite
//!   value the contract gives: 8128, and 2^-33 where two steps cancel and
//!   only the fused F64 multiply-add keeps the residual; a captured table
//!   and GEMM replayed.
//! - `g32_nvfp4_routed`: the same planes under routed tables — a 16-expert
//!   stack of 272 rows at K 4096 each slot its own column, an 8-expert stack
//!   at K 8192 whose two slots a token share the token's column, each expert
//!   its own tensor scale, uniform and same-expert routings at 1, 9, 64 and
//!   448 slots, a rerun bit-identical; an id past the stack
//!   raises `ExpertId`, its slots refused and NaN, the rest the clean run's
//!   bits.
//! - `g32_nvfp4_swiglu`: `enqueue_up_swiglu32`'s fold over NVFP4 planes
//!   (`gemm_nvfp4p_swiglu`: the up GEMM's products through `silu(g) · u`
//!   into the down's activations, never stored) against its two launches,
//!   `gemm_nvfp4p` into the up rows and then `swiglu_quant32_sel`: 16 card
//!   experts of 640 rows (five full slabs) and of 160 (a slab of 32 live
//!   rows) at K 512, under a remapped route that leaves every third of 24
//!   experts to the host, four slots a token sharing its column, at 1, 9, 64
//!   and 448 tokens: both sides seeded alike, every column's planes the two
//!   launches' bit for bit (the host columns untouched) and no fault; from
//!   64 tokens token 0's column lies far past the F32 walk's range, so its
//!   tiles walk again exactly (an up value past that range before the tensor
//!   scale), with the same bits; at 64 tokens a NaN gate value, a refused
//!   slot on a card place, a refused slot on a place no side serves and a
//!   listed slot on such a place give both sides the same planes and the
//!   same `ExpertId` and `QuantColumn` fault; each run's fold captured into
//!   a graph of one node whose replay over a fresh seed gives the eager
//!   planes and fault; an all-host route with one expert and 1 or 257 slots
//!   leaves host places untouched and raises `ExpertId` for an invalid last
//!   place in both eager and graph runs; and activations of another K than
//!   the up rows or of fewer columns than the slots, and a short gate or
//!   places buffer,
//!   refused by name before any launch. `--case nvfp4_swiglu` runs it
//!   alone, without the model file.
//! - `g32_swiglu_epi`: the same SwiGLU epilogue (`swiglu32.rs`) on the other
//!   up stacks that declare it, each against its up GEMM and quantizer:
//!   Q4_K through the grouped GEMM (`gemm_q4k_swiglu` against `gemm_q4k`
//!   and `swiglu_quant32_sel`) under `g32_nvfp4_swiglu`'s remapped route,
//!   shapes, token counts and fault cases, token 0's column from 64 tokens
//!   scaled under the F32 walk's activation scale floor so its tiles walk
//!   again exactly; and Q8_0 planes over a dense one-expert table, every
//!   slot (`gemm_q8_0p_swiglu_all` at 9 columns and its slab twin at a wave
//!   of the slab entry, against `gemm_q8_0p` and `swiglu_quant32`), column 0
//!   under the scale floor, clean, with a NaN gate value, and with a slot
//!   refused by an out-of-range expert id (the route's own `ExpertId` read
//!   apart, each side's NaN row raising `QuantColumn`). Every side's
//!   planes and fault bit for bit the two launches', each graph of one node
//!   replaying the eager run. `--case swiglu_epi` runs it alone, without
//!   the model file.
//! - `q5_0p_is_the_gemv`: the resident packed Q5_0 (`pack_q5_0`, the tensor
//!   the gemv reads) in `gemm_q5_0p` over the dense table, at Nemotron's
//!   `ssm_in` (17504 × 3136) and `ffn_up` (12544 × 3136) and at 272 rows of
//!   K 1056 (33 blocks), for U ∈ {1, 8, 9, 64, 512} columns, against
//!   `enqueue_gemv_q5_0` run eight columns at a time on the same input: both
//!   quantizers' planes the host's, the decode ggml's, every output of both
//!   written and finite, every checked output bit for bit the contract's
//!   transcription and within the gemv band — not bit-equal, the gemv sums
//!   a row's blocks in 32 lane partials and a butterfly, so
//!   `|y − y_gemv| <= γ(K/32 + ceil(K/1024) + 6) · Σ_b |d_a·d_w·isum_b|`
//!   (`gemm32::q5_0p_gemv_roundings` derives it).
//! - `g38_iq4nl`: the IQ4_NL down of the Qwen3.8 card route (`gemm_iq4nl`)
//!   — a synthetic 64 × 2560-row stack at K 640 (random nibble codes, finite
//!   f16 `d`), each slot its own column, under the four routings of the
//!   K-quant cases' token counts; its decode against ggml's `dequant_row`,
//!   a NaN `d` block's outputs NaN, a rerun bit-identical, and the host
//!   API's refusal of the stack at another K.
//! - `g32_remap`: `enqueue_route_remap` over 48 experts, every third on a
//!   16-expert card stack in reversed slots and the rest `HOST`: the listed
//!   slots and tiles the host grouping of the mapped ids less the host slots,
//!   no slot refused, every host slot's rows untouched (still the sentinel)
//!   and every card slot's rows the contract's bits; `gemm_q4k` over the same
//!   table leaves the host slots untouched and gives every card slot the bits
//!   it gives over `gemm_route`'s table of the mapped ids; an identity map
//!   gives `gemm_route`'s table; an id past the map and a map value past the stack
//!   each raise `ExpertId`, their slots refused and NaN, the rest the clean
//!   run's bits.
//! - `g32_f32tile`: `enqueue_f32_tile` bit for bit `enqueue_f32_gemv` over
//!   chunks of up to eight columns at 96, 100, 128 and 512 rows, up to 4095
//!   columns, nothing written past its output.
//! - `dense_mma`: the dense prompt MMA of a compute capability 12 device for
//!   BF16, MXFP8 and FP8_B128 weights (`enqueue_bf16_mma`,
//!   `enqueue_e4m3_mma`) at ragged shapes (one
//!   with columns past a raster group) and range edges: every output within
//!   the BF16 operand band, the cells of an edge's row and column the exact
//!   tile's bits, nothing written past the output and a graph replay the
//!   eager bits; the BF16 form's product the same bits eager and replayed
//!   over a form rewritten from zeros after the capture. At 130 × 130
//!   outputs of K 128, unit weights against activations of 1 + 2^-10: every
//!   cell K, and with a subnormal activation in one column, then also weights
//!   of 2^60 in one row, that column's and row's cells the exact tile's bits
//!   and every other cell still K. A BF16 weight off a 16-byte boundary is
//!   refused by name.
//! - `mxfp8_w8a8`: the native W8A8 MXFP8 product over the activations' MXFP8
//!   form (`enqueue_mxfp8_cols`, `enqueue_mxfp8_w8a8`): the form against a
//!   host quantizer, the product against an f64 oracle at ragged shapes, a
//!   graph replay over a form rewritten from zeros after the capture equal to
//!   the eager bits, an input value or a weight scale byte the tensor cores
//!   cannot take sending its column or row to the exact tile, and stale or
//!   misplaced forms refused by name.
//! - `zgate`: the BF16 and (native build) MXFP8 W8A8 projections' sigmoid
//!   norm epilogue against the separate product and norm, bit for bit at
//!   M=1/8/9/129/4095, including exact replay of a refused column or weight
//!   row, non-finite inputs, graph replay, sentinel tails and a token-unit cut.
//!   `--bench-kernels --bench-arm zgate` prices both arms of each form at 48
//!   heads of 128, K=2560 and M=4095 through the shared burst and graph timing
//!   owner.
//! - `g32_refuse`: the host API's refusals, each an error holding its own
//!   message (another check refusing first does not pass): the scratch's
//!   K and columns, the quantizer's input and columns, an unfilled table,
//!   rows not a multiple of 16, a layout or planes of another shape, an
//!   activation of fewer columns than the slots or quantized for fewer, packed
//!   Q5_0 rows of another K, NVFP4 planes at a K not a multiple of 64, a
//!   short `qs` or `e`, a tensor scale count other than the table's experts, a
//!   short `y`, a clamp limit that is not finite and a short input for the
//!   rule-taking SwiGLU, an empty map and slots past the table for the remap, and the
//!   F32 tile's K, short `x` and an `x` off a 16-byte boundary.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemm", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/bench.rs"]
mod bench_arm;

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::gemm::{
        GEMM_BN, GEMM_MAX_SLOTS, GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmTile,
        GemmWeight,
    };
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::iq::{IqFormat, IqKernels, IqRows};
    use bloomery_gpu::kquant::{Act, act::SWIGLU_OAI_ALPHA};
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::route_softmax::RouterDims;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::act_rule::q8_1_codes;
    use bloomery_gpu_gates::iq_rows::{ref_dequant, set_d};
    use bloomery_gpu_gates::rounding::{gamma, scaled_sum};
    use bloomery_gpu_gates::swiglu_host;
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, activations, bits_equal, bytes_to_words, checks_failed, data_dir,
        open_model, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::iq_tables::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
        KSIGNS_IQ2XS, KVALUES_IQ4NL,
    };
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use model::arch::models::shape::{MoeShape, rules};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// What `y` holds before a launch, so an output the kernel leaves alone
    /// reads back as these bits.
    const SENT: f32 = 1.0e30;
    /// Token counts per case.
    const TOKENS: [usize; 8] = [1, 15, 16, 17, 64, 511, 512, UBATCH];
    /// Slot counts up to which every row of every slot is checked.
    const ALL_ROWS_UP_TO: usize = 136;

    #[derive(Clone, Copy, Debug)]
    enum Routing {
        Uniform,
        SameTopk,
        Quarter,
        RoundRobin,
    }

    const ROUTINGS: [Routing; 4] = [
        Routing::Uniform,
        Routing::SameTopk,
        Routing::Quarter,
        Routing::RoundRobin,
    ];

    /// A 64-bit LCG, the gate's one source of synthetic values.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// The ids of `n_tok` tokens × `top_k` distinct experts of `n_exp` under
    /// routing `r`, slot `t·top_k + k`.
    fn route_ids(r: Routing, n_tok: usize, top_k: usize, n_exp: usize, seed: u64) -> Vec<u32> {
        let mut g = Lcg(seed);
        let mut ids = Vec::with_capacity(n_tok * top_k);
        for t in 0..n_tok {
            let pool = match r {
                Routing::Quarter => (n_exp / 4).max(top_k),
                _ => n_exp,
            };
            let mut pick: Vec<u32> = Vec::with_capacity(top_k);
            for k in 0..top_k {
                let e = match r {
                    Routing::SameTopk => (7 * k + 3) % n_exp,
                    Routing::RoundRobin => (t * top_k + k) % n_exp,
                    Routing::Uniform | Routing::Quarter => loop {
                        let c = g.below(pool);
                        if !pick.contains(&(c as u32)) {
                            break c;
                        }
                    },
                };
                pick.push(e as u32);
            }
            ids.extend(pick);
        }
        ids
    }

    /// One stack under test: its type, bytes (the file's stream), geometry.
    struct Stack<'a> {
        name: String,
        ty: GemmWeight,
        bytes: std::borrow::Cow<'a, [u8]>,
        n_exp: usize,
        rows: usize,
        k: usize,
        input: Input,
    }

    /// The case's slot-to-column rule, with the top-k the routing uses.
    #[derive(Clone, Copy)]
    enum Input {
        Shared(usize),
        PerSlot(usize),
    }

    impl Input {
        fn top_k(self) -> usize {
            match self {
                Input::Shared(k) | Input::PerSlot(k) => k,
            }
        }
        fn gemm(self) -> GemmInput {
            match self {
                Input::Shared(top_k) => GemmInput::Shared { top_k },
                Input::PerSlot(_) => GemmInput::PerSlot,
            }
        }
        /// Activation columns `n_slots` slots read.
        fn cols(self, n_slots: usize) -> usize {
            match self {
                Input::Shared(k) => n_slots / k,
                Input::PerSlot(_) => n_slots,
            }
        }
        fn col_of(self, slot: usize) -> usize {
            match self {
                Input::Shared(k) => slot / k,
                Input::PerSlot(_) => slot,
            }
        }
    }

    /// One row decoded to the integers the kernel multiplies: per
    /// super-block `d`, `dmin`; per scale group of `grp` values its scale;
    /// per 32 values its min (Q4_K, Q5_K); per value its code.
    struct Dec {
        grp: usize,
        d: Vec<f64>,
        dmin: Vec<f64>,
        sc: Vec<i32>,
        mn: Vec<i32>,
        q: Vec<i8>,
    }

    fn f16(b: &[u8]) -> f64 {
        f64::from(half_to_f32(u16::from_le_bytes([b[0], b[1]])))
    }

    /// ggml's `get_scale_min_k4`.
    fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
        if j < 4 {
            (i32::from(q[j] & 63), i32::from(q[j + 4] & 63))
        } else {
            (
                i32::from((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4)),
                i32::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
            )
        }
    }

    /// Decode one row of `k` values of type `ty` (ggml's `dequantize_row_*`
    /// read as integers).
    fn decode(ty: GemmWeight, row: &[u8], k: usize) -> Dec {
        let n_sb = k / 256;
        let bpb = ty.block_bytes();
        let grp = match ty {
            GemmWeight::Q4K
            | GemmWeight::Q5K
            | GemmWeight::Iq2Xxs
            | GemmWeight::Iq3Xxs
            | GemmWeight::Iq3S
            | GemmWeight::Iq1S
            | GemmWeight::Iq4Xs => 32,
            GemmWeight::Q6K
            | GemmWeight::Q3K
            | GemmWeight::Iq2Xs
            | GemmWeight::Iq2S
            | GemmWeight::Iq1M => 16,
        };
        let mut dec = Dec {
            grp,
            d: Vec::with_capacity(n_sb),
            dmin: Vec::with_capacity(n_sb),
            sc: Vec::with_capacity(k / grp),
            mn: Vec::with_capacity(k / 32),
            q: Vec::with_capacity(k),
        };
        for sb in 0..n_sb {
            let b = &row[sb * bpb..(sb + 1) * bpb];
            match ty {
                GemmWeight::Q4K | GemmWeight::Q5K => {
                    dec.d.push(f16(&b[0..2]));
                    dec.dmin.push(f16(&b[2..4]));
                    for j in 0..8 {
                        let (s, m) = scale_min_k4(j, &b[4..16]);
                        dec.sc.push(s);
                        dec.mn.push(m);
                    }
                    for v in 0..256 {
                        let (p, h, l) = (v / 64, (v % 64) / 32, v % 32);
                        let q = if ty == GemmWeight::Q4K {
                            (b[16 + 32 * p + l] >> (4 * h)) & 0xf
                        } else {
                            ((b[48 + 32 * p + l] >> (4 * h)) & 0xf)
                                | (((b[16 + l] >> (2 * p + h)) & 1) << 4)
                        };
                        dec.q.push(q as i8);
                    }
                }
                GemmWeight::Q6K => {
                    dec.d.push(f16(&b[208..210]));
                    dec.dmin.push(0.0);
                    for g in 0..16 {
                        dec.sc.push(i32::from(b[192 + g] as i8));
                    }
                    for v in 0..256 {
                        let (n, c, l) = (v / 128, (v % 128) / 32, v % 32);
                        let ql = b[64 * n + l + 32 * (c & 1)];
                        let qh = b[128 + 32 * n + l];
                        let q = ((ql >> (4 * (c >> 1))) & 0xf) | (((qh >> (2 * c)) & 3) << 4);
                        dec.q.push(q as i8 - 32);
                    }
                }
                GemmWeight::Q3K => {
                    dec.d.push(f16(&b[108..110]));
                    dec.dmin.push(0.0);
                    let w = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
                    let (a0, a1, a2) = (w(96), w(100), w(104));
                    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
                    let aux = [
                        (a0 & k2) | ((a2 & k1) << 4),
                        (a1 & k2) | (((a2 >> 2) & k1) << 4),
                        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
                        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
                    ];
                    for g in 0..16 {
                        dec.sc
                            .push(((aux[g / 4] >> (8 * (g % 4))) & 0xff) as i32 - 32);
                    }
                    for v in 0..256 {
                        let (n, j, l) = (v / 128, (v % 128) / 32, v % 32);
                        let low = ((b[32 + 32 * n + l] >> (2 * j)) & 3) as i8;
                        let high = b[l] & (1 << (4 * n + j)) != 0;
                        dec.q.push(low - if high { 0 } else { 4 });
                    }
                }
                GemmWeight::Iq2Xxs => {
                    // The kernel's epilogue scales by d/8 and keeps `2s + 1`
                    // whole in the integer scale (iq.rs's rule); d/8 is exact
                    // in f64, so both the band and the contract read it.
                    dec.d.push(f16(&b[0..2]) / 8.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        let aux = u32::from_le_bytes([
                            b[6 + 8 * c],
                            b[7 + 8 * c],
                            b[8 + 8 * c],
                            b[9 + 8 * c],
                        ]);
                        dec.sc.push(2 * (aux >> 28) as i32 + 1);
                        let idx = |l: usize| usize::from(b[2 + 8 * c + l]);
                        let signs = |l: usize| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let g = i32::from(IQ2XXS_GRID[idx(l)].to_le_bytes()[j]);
                            let sgn = if signs(l) & KMASK_IQ2XS[j] != 0 {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq3Xxs => {
                    // The kernel's epilogue scales by d/4 and keeps `2s + 1`
                    // whole in the integer scale (iq.rs's rule); d/4 is exact
                    // in f64, so both the band and the contract read it.
                    dec.d.push(f16(&b[0..2]) / 4.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        let aux = u32::from_le_bytes([
                            b[66 + 4 * c],
                            b[67 + 4 * c],
                            b[68 + 4 * c],
                            b[69 + 4 * c],
                        ]);
                        dec.sc.push(2 * (aux >> 28) as i32 + 1);
                        let idx = |j: usize| usize::from(b[2 + 8 * c + j]);
                        let signs = |l: usize| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let (ib, m) = if j < 4 {
                                (idx(2 * l), j)
                            } else {
                                (idx(2 * l + 1), j - 4)
                            };
                            let g = i32::from(IQ3XXS_GRID[ib].to_le_bytes()[m]);
                            let sgn = if signs(l) & KMASK_IQ2XS[j] != 0 {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq3S => {
                    // The kernel keeps `2s + 1` whole in the integer scale and
                    // scales by `d` (the grid is unscaled), as ggml's
                    // `dequantize_row_iq3_s` forms `d·(1 + 2s)·grid`.
                    dec.d.push(f16(&b[0..2]));
                    dec.dmin.push(0.0);
                    let (qs, qh, signs, scales) =
                        (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
                    for c in 0..8 {
                        dec.sc
                            .push(2 * i32::from((scales[c / 2] >> (4 * (c % 2))) & 0x0f) + 1);
                        for v in 0..32 {
                            let m = v / 4;
                            let index =
                                usize::from(qs[8 * c + m]) | (usize::from(qh[c] >> m & 1) << 8);
                            let g = i32::from(IQ3S_GRID[index].to_le_bytes()[v % 4]);
                            let sgn = if signs[4 * c + v / 8] & KMASK_IQ2XS[v % 8] != 0 {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq2Xs => {
                    // The kernel keeps `2s + 1` whole in the integer scale and
                    // scales by d/8; ggml forms `d·(0.5 + s)·0.25·grid`, the
                    // same product with d/8 exact in f64. One scale a 16-value
                    // half of each 32-value group.
                    dec.d.push(f16(&b[0..2]) / 8.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        for half in 0..2 {
                            dec.sc
                                .push(2 * i32::from((b[66 + c] >> (4 * half)) & 0x0f) + 1);
                        }
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let code =
                                u16::from_le_bytes([b[2 + 8 * c + 2 * l], b[3 + 8 * c + 2 * l]]);
                            let g = i32::from(IQ2XS_GRID[usize::from(code & 511)].to_le_bytes()[j]);
                            let sgn = if KSIGNS_IQ2XS[usize::from(code >> 9)] & KMASK_IQ2XS[j] != 0
                            {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq2S => {
                    // As IQ2_XS in the scales; the grid index takes its top two
                    // bits from `qh` and the signs are the raw bytes after the
                    // grid bytes.
                    dec.d.push(f16(&b[0..2]) / 8.0);
                    dec.dmin.push(0.0);
                    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
                    for c in 0..8 {
                        for half in 0..2 {
                            dec.sc
                                .push(2 * i32::from((scales[c] >> (4 * half)) & 0x0f) + 1);
                        }
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let index = usize::from(qs[4 * c + l])
                                | ((usize::from(qh[c]) << (8 - 2 * l)) & 0x300);
                            let g = i32::from(IQ2S_GRID[index].to_le_bytes()[j]);
                            let sgn = if signs[4 * c + l] & KMASK_IQ2XS[j] != 0 {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq1S => {
                    // ggml forms `d·(2s + 1)·(grid ± 1/8)`; the kernel keeps the
                    // int8 weight `8·grid ± 1` and scales by d/8, the same
                    // product with d/8 exact in f64.
                    dec.d.push(f16(&b[0..2]) / 8.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        let qh = u16::from_le_bytes([b[34 + 2 * c], b[35 + 2 * c]]);
                        dec.sc.push(2 * i32::from((qh >> 12) & 7) + 1);
                        let delta = if qh & 0x8000 != 0 { -1 } else { 1 };
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let index = usize::from(b[2 + 4 * c + l])
                                | (usize::from((qh >> (3 * l)) & 7) << 8);
                            let g = i32::from(IQ1S_GRID[index].to_le_bytes()[j] as i8);
                            dec.q.push((8 * g + delta) as i8);
                        }
                    }
                }
                GemmWeight::Iq1M => {
                    // As IQ1_S in the weights, one scale a 16-value half of
                    // each 32-value group (three bits each of the four scale
                    // halfwords) and one delta sign an 8-value row of the
                    // grid; `d` is the f16 spread over the halfwords' top
                    // nibbles, the C's expression.
                    let sc: [u16; 4] =
                        std::array::from_fn(|i| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]));
                    let bits = (sc[0] >> 12)
                        | ((sc[1] >> 8) & 0x00f0)
                        | ((sc[2] >> 4) & 0x0f00)
                        | (sc[3] & 0xf000);
                    dec.d.push(f64::from(half_to_f32(bits)) / 8.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        let s16 = sc[c / 2] >> (6 * (c % 2));
                        dec.sc.push(2 * i32::from(s16 & 7) + 1);
                        dec.sc.push(2 * i32::from((s16 >> 3) & 7) + 1);
                        let (qh0, qh1) = (b[32 + 2 * c], b[33 + 2 * c]);
                        let idx = [
                            usize::from(b[4 * c]) | ((usize::from(qh0) << 8) & 0x700),
                            usize::from(b[4 * c + 1]) | ((usize::from(qh0) << 4) & 0x700),
                            usize::from(b[4 * c + 2]) | ((usize::from(qh1) << 8) & 0x700),
                            usize::from(b[4 * c + 3]) | ((usize::from(qh1) << 4) & 0x700),
                        ];
                        let delta = [
                            if qh0 & 0x08 != 0 { -1 } else { 1 },
                            if qh0 & 0x80 != 0 { -1 } else { 1 },
                            if qh1 & 0x08 != 0 { -1 } else { 1 },
                            if qh1 & 0x80 != 0 { -1 } else { 1 },
                        ];
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let g = i32::from(IQ1S_GRID[idx[l]].to_le_bytes()[j] as i8);
                            dec.q.push((8 * g + delta[l]) as i8);
                        }
                    }
                }
                GemmWeight::Iq4Xs => {
                    dec.d.push(f16(&b[0..2]));
                    dec.dmin.push(0.0);
                    let sh = u16::from_le_bytes([b[2], b[3]]);
                    for c in 0..8 {
                        let ls = (i32::from((b[4 + c / 2] >> (4 * (c % 2))) & 0x0f)
                            | (i32::from((sh >> (2 * c)) & 3) << 4))
                            - 32;
                        dec.sc.push(ls);
                        for j in 0..16 {
                            dec.q
                                .push(KVALUES_IQ4NL[usize::from(b[8 + 16 * c + j] & 0x0f)]);
                        }
                        for j in 0..16 {
                            dec.q
                                .push(KVALUES_IQ4NL[usize::from(b[8 + 16 * c + j] >> 4)]);
                        }
                    }
                }
            }
        }
        dec
    }

    /// Whether `dec` reproduces ggml's `dequant_row` of the same row: every
    /// value within four f32 roundings of its two terms' magnitude (ggml
    /// forms `d·sc` and `dmin·m` in f32 and then the value).
    fn decode_matches(ty: GgmlType, row: &[u8], dec: &Dec) -> Result<bool, GateError> {
        let k = dec.q.len();
        let mut want = vec![0.0f32; k];
        match ty {
            // `dequant_row` has no decode for these three: the gate's own
            // port of ggml's `dequantize_row_*`.
            GgmlType::IQ2_S => ref_dequant(IqFormat::Iq2S, row, &mut want)?,
            GgmlType::IQ1_S => ref_dequant(IqFormat::Iq1S, row, &mut want)?,
            GgmlType::IQ1_M => ref_dequant(IqFormat::Iq1M, row, &mut want)?,
            _ => dequant_row(ty, row, &mut want)?,
        }
        Ok((0..k).all(|v| {
            let sb = v / 256;
            let a = dec.d[sb] * f64::from(dec.sc[v / dec.grp]) * f64::from(dec.q[v]);
            let m = if dec.mn.is_empty() {
                0.0
            } else {
                dec.dmin[sb] * f64::from(dec.mn[v / 32])
            };
            ((a - m) - f64::from(want[v])).abs() <= gamma(4) * (a.abs() + m.abs()) + 1e-38
        }))
    }

    /// One output three ways, summed per 128-value block as the kernel groups
    /// it: the f64 reference, its band's magnitude `Σ_b (|D_b| + |M_b|)`, and
    /// the module's numeric contract transcribed — per block the exact i32
    /// `isum` and `imin`, then `acc = fma(d8, fma(−dmin, f32(imin),
    /// d·f32(isum)), acc)` (`acc = fma(d8, d·f32(isum), acc)` for a type
    /// without mins) from `acc = 0`, blocks in increasing k, each step a
    /// [`scaled_sum`] into the F32 accumulator. `d` and `dmin` are f16
    /// values, exact in f32.
    fn dot_ref(dec: &Dec, a: &[i8], d8: &[f32]) -> (f64, f64, f32) {
        let (mut y, mut mag) = (0.0f64, 0.0f64);
        let mut acc = 0.0f32;
        for (b, &db) in d8.iter().enumerate() {
            let sb = b / 2;
            let mut isum = 0i32;
            for g in (128 * b..128 * b + 128).step_by(dec.grp) {
                let dot: i32 = (g..g + dec.grp)
                    .map(|v| i32::from(dec.q[v]) * i32::from(a[v]))
                    .sum();
                isum += dec.sc[g / dec.grp] * dot;
            }
            let mut imin = 0i32;
            if !dec.mn.is_empty() {
                for j in (128 * b..128 * b + 128).step_by(32) {
                    let s: i32 = a[j..j + 32].iter().map(|&v| i32::from(v)).sum();
                    imin += dec.mn[j / 32] * s;
                }
            }
            let (nd, m) = if dec.mn.is_empty() {
                (0.0, 0.0)
            } else {
                (-(dec.dmin[sb] as f32), imin as f32)
            };
            acc = scaled_sum(
                [f64::from(dec.d[sb] as f32), f64::from(nd)],
                [isum as f32, m],
                db,
                f64::from(acc),
            ) as f32;
            let dd = f64::from(db) * dec.d[sb] * f64::from(isum);
            let mm = f64::from(db) * dec.dmin[sb] * f64::from(imin);
            y += dd - mm;
            mag += dd.abs() + mm.abs();
        }
        (y, mag, acc)
    }

    /// Whether row `r` of slot `s` of an `n_slots`-slot run is checked
    /// against the reference: every row up to [`ALL_ROWS_UP_TO`] slots, a
    /// quarter of them above, rotating with the slot.
    fn row_checked(n_slots: usize, s: usize, r: usize) -> bool {
        n_slots <= ALL_ROWS_UP_TO || (r + s).is_multiple_of(4)
    }

    /// The reference check of one run: the first output outside its band,
    /// the largest error-to-band ratio, the count checked; the outputs whose
    /// bits differ from the contract's transcription and the first of them;
    /// and whether every decoded row matched ggml.
    struct RefOutcome {
        fail: Option<String>,
        worst_ratio: f64,
        checked: usize,
        bits_differ: usize,
        bits_fail: Option<String>,
        decode_ok: bool,
    }

    impl RefOutcome {
        fn new() -> RefOutcome {
            RefOutcome {
                fail: None,
                worst_ratio: 0.0,
                checked: 0,
                bits_differ: 0,
                bits_fail: None,
                decode_ok: true,
            }
        }
    }

    /// Check `y` (slot-major, `rows` per slot) against the f64 reference for
    /// every slot and the checked rows, in parallel over experts.
    fn check_ref(
        st: &Stack<'_>,
        ids: &[u32],
        codes: &[Vec<i8>],
        d8: &[Vec<f32>],
        y: &[f32],
    ) -> Result<RefOutcome, GateError> {
        let n_slots = ids.len();
        let mut by_exp: Vec<Vec<usize>> = vec![Vec::new(); st.n_exp];
        for (s, &e) in ids.iter().enumerate() {
            by_exp[e as usize].push(s);
        }
        // Work items are (expert, 64-row block), so a routing that puts every
        // slot on a few experts still spreads over every thread.
        let work: Vec<(usize, usize)> = (0..st.n_exp)
            .filter(|&e| !by_exp[e].is_empty())
            .flat_map(|e| (0..st.rows.div_ceil(64)).map(move |b| (e, b)))
            .collect();
        let next = AtomicUsize::new(0);
        let rb = st.ty.block_bytes() * st.k / 256;
        let ggml_ty = match st.ty {
            GemmWeight::Q3K => GgmlType::Q3_K,
            GemmWeight::Q4K => GgmlType::Q4_K,
            GemmWeight::Q5K => GgmlType::Q5_K,
            GemmWeight::Q6K => GgmlType::Q6_K,
            GemmWeight::Iq2Xxs => GgmlType::IQ2_XXS,
            GemmWeight::Iq2Xs => GgmlType::IQ2_XS,
            GemmWeight::Iq2S => GgmlType::IQ2_S,
            GemmWeight::Iq1S => GgmlType::IQ1_S,
            GemmWeight::Iq1M => GgmlType::IQ1_M,
            GemmWeight::Iq3Xxs => GgmlType::IQ3_XXS,
            GemmWeight::Iq3S => GgmlType::IQ3_S,
            GemmWeight::Iq4Xs => GgmlType::IQ4_XS,
        };
        let nb = st.k / 128;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
        let results: Vec<Result<RefOutcome, String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut out = RefOutcome::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(e, blk)) = work.get(i) else { break };
                            for r in 64 * blk..(64 * blk + 64).min(st.rows) {
                                let off = (e * st.rows + r) * rb;
                                let row = &st.bytes[off..off + rb];
                                let dec = decode(st.ty, row, st.k);
                                out.decode_ok &=
                                    decode_matches(ggml_ty, row, &dec).map_err(|x| x.to_string())?;
                                for &s in by_exp[e].iter().filter(|&&s| row_checked(n_slots, s, r)) {
                                    let c = st.input.col_of(s);
                                    let (want, mag, contract) = dot_ref(&dec, &codes[c], &d8[c]);
                                    let got = y[s * st.rows + r];
                                    let band = gamma(nb + 4) * mag;
                                    let err = (f64::from(got) - want).abs();
                                    let ratio = if band > 0.0 { err / band } else { err };
                                    out.checked += 1;
                                    let within = err <= band;
                                    if !within && out.fail.is_none() {
                                        out.fail = Some(format!(
                                            "slot {s} row {r} expert {e}: got {got:e} want {want:e} \
                                             err {err:.3e} band {band:.3e}"
                                        ));
                                    }
                                    out.worst_ratio = out.worst_ratio.max(ratio);
                                    if got.to_bits() != contract.to_bits() {
                                        out.bits_differ += 1;
                                        if out.bits_fail.is_none() {
                                            out.bits_fail = Some(format!(
                                                "slot {s} row {r} expert {e}: got {got:e} ({:#010x}) \
                                                 contract {contract:e} ({:#010x})",
                                                got.to_bits(),
                                                contract.to_bits()
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        Ok(out)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("reference thread panicked".into()))
                })
                .collect()
        });
        let mut all = RefOutcome::new();
        for r in results {
            let r = r?;
            all.fail = all.fail.or(r.fail);
            all.worst_ratio = all.worst_ratio.max(r.worst_ratio);
            all.checked += r.checked;
            all.bits_differ += r.bits_differ;
            all.bits_fail = all.bits_fail.or(r.bits_fail);
            all.decode_ok &= r.decode_ok;
        }
        Ok(all)
    }

    /// The device side of one run: activations quantized, the route table,
    /// the GEMM; `y` read back.
    struct Dev<'g> {
        gpu: &'g Gpu,
        gk: GemmKernels,
        q6s: Q6kSelKernels,
        iqk: IqKernels,
    }

    /// The case's resident stack and scratch.
    struct Resident {
        w: DeviceTensor<u32>,
        act: GemmAct,
        route: GemmRoute,
        ids: DeviceBuffer<u32>,
        y: DeviceBuffer<f32>,
        /// `Q8Act`s of 1..=8 columns for the gemv diagnostic.
        acts: Vec<Q8Act>,
    }

    impl Dev<'_> {
        fn resident(&self, st: &Stack<'_>) -> Result<Resident, GateError> {
            self.resident_at(st, TOKENS[TOKENS.len() - 1] * st.input.top_k())
        }

        /// [`Self::resident`] for a case whose largest run is `max_slots`
        /// slots.
        fn resident_at(&self, st: &Stack<'_>, max_slots: usize) -> Result<Resident, GateError> {
            let stream = self.gpu.stream();
            let n_rows = st.n_exp * st.rows;
            let mut words = bytes_to_words(&st.bytes);
            words.resize(words.len().div_ceil(n_rows) * n_rows, 0);
            let cols = words.len() / n_rows;
            let acts = (1..=8)
                .map(|m| Q8Act::with_k(stream, m, st.k))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Resident {
                w: DeviceTensor::upload(stream, &words, n_rows, cols)?,
                act: GemmAct::new(stream, st.input.cols(max_slots), st.k)?,
                route: GemmRoute::new(stream, max_slots, st.n_exp)?,
                ids: DeviceBuffer::zeroed(stream, max_slots)?,
                y: DeviceBuffer::zeroed(stream, max_slots * st.rows)?,
                acts,
            })
        }

        /// Quantize `x` (`n_cols` columns), route `ids`, run the GEMM; `y`
        /// starts at [`SENT`]. Returns `y`'s first `ids.len()·rows` values.
        fn run(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &DeviceBuffer<f32>,
            ids: &[u32],
        ) -> Result<Vec<f32>, GateError> {
            let stream = self.gpu.stream();
            let n_slots = ids.len();
            res.ids.copy_from_host(stream, &pad(ids, res.ids.len()))?;
            res.y.copy_from_host(stream, &vec![SENT; res.y.len()])?;
            self.gpu.enqueue_quantize_gemm(
                x,
                st.input.cols(n_slots),
                &mut res.act,
                self.gpu.unlabelled_sink(),
            )?;
            self.gk.enqueue_route(
                stream,
                &res.ids,
                n_slots,
                &mut res.route,
                self.gpu.unlabelled_sink(),
            )?;
            self.gk.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: st.ty,
                    w: &res.w,
                    rows_per_expert: st.rows,
                    act: &res.act,
                    route: &res.route,
                    input: st.input.gemm(),
                    y: &mut res.y,
                },
            )?;
            stream.synchronize()?;
            let mut y = res.y.to_host_vec(stream)?;
            y.truncate(n_slots * st.rows);
            Ok(y)
        }

        /// Today's `_sel` gemv over the same codes, slot-major like `y`: up to
        /// eight slots a launch, each slot's column quantized into its own
        /// `Q8Act` column. `None` for a type with no `_sel` kernel.
        fn gemv_ref(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &[f32],
            ids: &[u32],
        ) -> Result<Option<Vec<f32>>, GateError> {
            let stream = self.gpu.stream();
            let n_slots = ids.len();
            let mut out = vec![0.0f32; n_slots * st.rows];
            let mut ybuf = DeviceBuffer::<f32>::zeroed(stream, 8 * st.rows)?;
            match (st.ty, st.input) {
                (GemmWeight::Q5K, _) | (GemmWeight::Q3K, Input::PerSlot(_)) => return Ok(None),
                // The i-quant types' diagnostic is `iq_rows_diag`, not a
                // `_sel` gemv.
                (GemmWeight::Iq2Xxs, _)
                | (GemmWeight::Iq2Xs, _)
                | (GemmWeight::Iq2S, _)
                | (GemmWeight::Iq1S, _)
                | (GemmWeight::Iq1M, _)
                | (GemmWeight::Iq3Xxs, _)
                | (GemmWeight::Iq3S, _)
                | (GemmWeight::Iq4Xs, _) => {
                    return Ok(None);
                }
                (GemmWeight::Q3K, Input::Shared(top_k)) => {
                    // One launch per token: every slot of it reads the one column.
                    for t in 0..n_slots / top_k {
                        let xc = DeviceBuffer::from_host(stream, &x[t * st.k..(t + 1) * st.k])?;
                        let act = &mut res.acts[0];
                        self.gpu.enqueue_quantize_q8_1(&xc, act)?;
                        let sel =
                            DeviceBuffer::from_host(stream, &ids[t * top_k..(t + 1) * top_k])?;
                        self.gpu
                            .enqueue_gemv_q3k_sel(&res.w, act, &sel, top_k, st.rows, &mut ybuf)?;
                        let got = ybuf.to_host_vec(stream)?;
                        out[t * top_k * st.rows..(t + 1) * top_k * st.rows]
                            .copy_from_slice(&got[..top_k * st.rows]);
                    }
                }
                (GemmWeight::Q4K | GemmWeight::Q6K, _) => {
                    for s0 in (0..n_slots).step_by(8) {
                        let m = (n_slots - s0).min(8);
                        let mut xs = Vec::with_capacity(m * st.k);
                        for s in s0..s0 + m {
                            let c = st.input.col_of(s);
                            xs.extend_from_slice(&x[c * st.k..(c + 1) * st.k]);
                        }
                        let xc = DeviceBuffer::from_host(stream, &xs)?;
                        let act = &mut res.acts[m - 1];
                        self.gpu.enqueue_quantize_q8_1(&xc, act)?;
                        let sel = DeviceBuffer::from_host(stream, &ids[s0..s0 + m])?;
                        if st.ty == GemmWeight::Q4K {
                            self.gpu.q4k_sel().enqueue_gemv_q4k_sel(
                                stream, &res.w, act, &sel, m, st.rows, &mut ybuf,
                            )?;
                        } else {
                            self.q6s.enqueue_gemv_q6k_sel(
                                stream, &res.w, act, &sel, m, st.rows, &mut ybuf,
                            )?;
                        }
                        let got = ybuf.to_host_vec(stream)?;
                        out[s0 * st.rows..(s0 + m) * st.rows].copy_from_slice(&got[..m * st.rows]);
                    }
                }
            }
            Ok(Some(out))
        }
    }

    impl Dev<'_> {
        /// One `_sel` gemv launch of `ids` over `res.w` into a [`SENT`]-filled
        /// `y`: Q3_K's slots share one column of `x`, Q4_K's and Q6_K's slot
        /// `s` reads column `s`. Returns `y`.
        fn sel_once(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &[f32],
            ids: &[u32],
        ) -> Result<Vec<f32>, GateError> {
            let stream = self.gpu.stream();
            let m = ids.len();
            let sel = DeviceBuffer::from_host(stream, ids)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; m * st.rows])?;
            let cols = if st.ty == GemmWeight::Q3K { 1 } else { m };
            let xc = DeviceBuffer::from_host(stream, &x[..cols * st.k])?;
            let act = &mut res.acts[cols - 1];
            self.gpu.enqueue_quantize_q8_1(&xc, act)?;
            match st.ty {
                GemmWeight::Q3K => self
                    .gpu
                    .enqueue_gemv_q3k_sel(&res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q4K => self
                    .gpu
                    .q4k_sel()
                    .enqueue_gemv_q4k_sel(stream, &res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q6K => self
                    .q6s
                    .enqueue_gemv_q6k_sel(stream, &res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q5K => return Err("Q5_K has no _sel gemv".into()),
                GemmWeight::Iq2Xxs
                | GemmWeight::Iq2Xs
                | GemmWeight::Iq2S
                | GemmWeight::Iq1S
                | GemmWeight::Iq1M
                | GemmWeight::Iq3Xxs
                | GemmWeight::Iq3S
                | GemmWeight::Iq4Xs => {
                    return Err("an i-quant stack has no _sel gemv".into());
                }
            }
            stream.synchronize()?;
            Ok(y.to_host_vec(stream)?)
        }
    }

    /// The `_sel` gemvs' out-of-range ids, per kernel (Q3_K, Q4_K, Q6_K) on
    /// a synthetic 16-expert stack: slot 1 at three past the stack raises
    /// [`FaultSite::ExpertId`] as an unlabelled launch, slot 1 at [`HOST`]
    /// (a slot the host tier serves) raises nothing; both leave slot 1 at
    /// [`SENT`], and every other slot bit for bit what the in-range launch
    /// writes.
    fn sel_faults(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let (k, rows, n_exp) = (2048usize, 256usize, 16usize);
        let want_id = Fault::at(LAYER_NONE, FaultSite::ExpertId);
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let mut ok = true;
        for (ty, input, seed) in [
            (GemmWeight::Q3K, Input::Shared(4), 0x5e13),
            (GemmWeight::Q4K, Input::PerSlot(4), 0x5e14),
            (GemmWeight::Q6K, Input::PerSlot(4), 0x5e16),
        ] {
            let st = Stack {
                name: format!("sel_fault_{ty:?}"),
                ty,
                bytes: synthetic(ty, n_exp * rows * k / 256, seed).into(),
                n_exp,
                rows,
                k,
                input,
            };
            let mut res = dev.resident(&st)?;
            let x = activations(k, 4, u32::try_from(seed)?);
            let clean = gpu.take_fault()?;
            let good = dev.sel_once(&st, &mut res, &x, &[3, 5, 0, 12])?;
            let good_fault = gpu.take_fault()?;
            let mut pass = clean.is_none() && good_fault.is_none();
            for (case, bad, want) in [
                ("past_stack", n_exp as u32 + 3, Some(want_id)),
                ("host", HOST, None),
            ] {
                let y = dev.sel_once(&st, &mut res, &x, &[3, bad, 0, 12])?;
                let fault = gpu.take_fault()?;
                let untouched = y[rows..2 * rows]
                    .iter()
                    .all(|v| v.to_bits() == SENT.to_bits());
                let others = [0usize, 2, 3].iter().all(|&s| {
                    bits_equal(
                        &y[s * rows..(s + 1) * rows],
                        &good[s * rows..(s + 1) * rows],
                    )
                });
                let case_ok = fault == want && untouched && others;
                println!(
                    "sel fault={case} ty={ty:?} sel=[3, {bad}, 0, 12] fault=\"{}\" want=\"{}\" \
                     slot1_untouched={untouched} other_slots_bit_identical={others} {}",
                    shown(fault),
                    shown(want),
                    verdict(case_ok)
                );
                pass &= case_ok;
            }
            if clean.is_some() || good_fault.is_some() {
                println!(
                    "sel fault=clean ty={ty:?} word_before={} after_in_range={} (want none) {}",
                    shown(clean),
                    shown(good_fault),
                    verdict(false)
                );
            }
            ok &= pass;
        }
        Ok(ok)
    }

    /// The fault's site mask: three launches raise, the first in time on
    /// layer 9 (`norm_quant` over a NaN column: [`FaultSite::NormQuant`]),
    /// then two on layer 3 (the GEMM quantizer over a NaN column,
    /// [`FaultSite::QuantColumn`], and the route over an id past the stack,
    /// [`FaultSite::ExpertId`]). The fault names layer 3, the smaller code
    /// there and a mask of exactly those two sites — layer 9's site is not in
    /// it — both read from the card's words and through the argmax copy the
    /// head's readback makes (token, word, mask).
    fn site_mask(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let (k, rows, n_exp, top_k) = (2048usize, 256usize, 16usize, 4usize);
        let st = Stack {
            name: "site_mask".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x0fa3).into(),
            n_exp,
            rows,
            k,
            input: Input::Shared(top_k),
        };
        let mut res = dev.resident(&st)?;
        let clean = gpu.take_fault()?;
        // Layer 9 first: a later layer raised earlier in time.
        let mut xn = activations(k, 1, 997);
        xn[11] = f32::NAN;
        let xn = DeviceBuffer::from_host(stream, &xn)?;
        let gain = DeviceBuffer::from_host(stream, &vec![1.0f32; k])?;
        let mut normed = DeviceBuffer::<f32>::zeroed(stream, k)?;
        let fused = bloomery_gpu::fused::FusedKernels::load(gpu.context())?;
        fused.enqueue_norm_quant(
            stream,
            &xn,
            &gain,
            1e-6,
            &mut res.acts[0],
            &mut normed,
            gpu.layer_sink(9)?,
        )?;
        // Then layer 3, twice.
        let n_tok = 2;
        let mut ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 29);
        ids[2] = n_exp as u32 + 1;
        let mut x = activations(k, n_tok, 999);
        x[k + 5] = f32::NAN;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        res.ids.copy_from_host(stream, &pad(&ids, res.ids.len()))?;
        let l3 = gpu.layer_sink(3)?;
        gpu.enqueue_quantize_gemm(&xd, st.input.cols(ids.len()), &mut res.act, l3)?;
        dev.gk
            .enqueue_route(stream, &res.ids, ids.len(), &mut res.route, l3)?;
        // The head's copy of the pair.
        let logits = DeviceBuffer::from_host(stream, &activations(64, 1, 1001))?;
        let mut out = DeviceBuffer::from_host(stream, &[7u32; 3])?;
        gpu.elem()
            .enqueue_argmax_fault(stream, &logits, 64, gpu.unlabelled_sink(), &mut out)?;
        let out = out.to_host_vec(stream)?;
        let copied = Fault::from_words(out[1], out[2]);
        let fault = gpu.take_fault()?;
        let want = Fault::of_sites(3, &[FaultSite::ExpertId, FaultSite::QuantColumn]);
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let masks = |f: Option<Fault>| f.map_or(0, |f| f.sites);
        let pass = clean.is_none() && fault == Some(want) && copied == Some(want);
        println!(
            "gemm fault=site_mask layer9=[norm_quant] then layer3=[quant_column, expert_id]: \
             word=\"{}\" mask={:#06x} copy=\"{}\" mask={:#06x} want=\"{want}\" mask={:#06x} \
             word_before={} {}",
            shown(fault),
            masks(fault),
            shown(copied),
            masks(copied),
            want.sites,
            shown(clean),
            verdict(pass)
        );
        Ok(pass)
    }

    /// `v` zero-padded to `n` entries.
    fn pad(v: &[u32], n: usize) -> Vec<u32> {
        let mut p = v.to_vec();
        p.resize(n, 0);
        p
    }

    /// `max|a − b| / max|b|`.
    fn rel_diff(a: &[f32], b: &[f32]) -> f64 {
        let num = a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
            m.max((f64::from(x) - f64::from(y)).abs())
        });
        let den = b.iter().fold(0.0f64, |m, &y| m.max(f64::from(y).abs()));
        if den > 0.0 { num / den } else { num }
    }

    /// The `--case` filter: `None` runs every case.
    fn case_filter() -> Option<String> {
        let mut args = std::env::args().skip_while(|a| a != "--case");
        args.next().and(args.next())
    }

    /// Whether the case named `name` runs under the filter.
    fn wanted(name: &str) -> bool {
        case_filter().is_none_or(|f| name.contains(&f))
    }

    /// Every case: T × routing runs, then the rerun and graph checks.
    fn run_case(dev: &Dev<'_>, st: &Stack<'_>, seed: u64) -> Result<bool, GateError> {
        if !wanted(&st.name) {
            return Ok(true);
        }
        let t0 = std::time::Instant::now();
        let stream = dev.gpu.stream();
        let mut res = dev.resident(st)?;
        let top_k = st.input.top_k();
        let mut ok = true;
        for &n_tok in &TOKENS {
            for (ri, &r) in ROUTINGS.iter().enumerate() {
                let n_slots = n_tok * top_k;
                let ids = route_ids(
                    r,
                    n_tok,
                    top_k,
                    st.n_exp,
                    seed ^ (n_tok as u64) << 8 ^ ri as u64,
                );
                let n_cols = st.input.cols(n_slots);
                let x = activations(
                    st.k,
                    n_cols,
                    (seed as u32) ^ (n_tok as u32 * 131 + ri as u32),
                );
                let xd = DeviceBuffer::from_host(stream, &x)?;
                let y = dev.run(st, &mut res, &xd, &ids)?;
                let unwritten = y.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                // The device's scales and sums against the host transcription:
                // the codes the reference uses are the kernel's.
                let (codes, d8): (Vec<Vec<i8>>, Vec<Vec<f32>>) = (0..n_cols)
                    .map(|c| q8_1_codes(&x[c * st.k..(c + 1) * st.k]))
                    .unzip();
                let dev_d8 = res.act.d8().to_host_vec(stream)?;
                let dev_s8 = res.act.s8().to_host_vec(stream)?;
                let nb = st.k / 128;
                let act_ok = (0..n_cols).all(|c| {
                    let d8_ok = (0..nb).all(|b| dev_d8[c * nb + b].to_bits() == d8[c][b].to_bits());
                    let s8_ok = (0..st.k / 32).all(|j| {
                        let s: i32 = codes[c][32 * j..32 * j + 32]
                            .iter()
                            .map(|&v| i32::from(v))
                            .sum();
                        dev_s8[c * (st.k / 32) + j] == s
                    });
                    d8_ok && s8_ok
                });
                let refc = check_ref(st, &ids, &codes, &d8, &y)?;
                let gemv = dev.gemv_ref(st, &mut res, &x, &ids)?;
                let diag = gemv.map_or_else(
                    || "none".to_string(),
                    |g| format!("{:.3e}", rel_diff(&y, &g)),
                );
                let pass = unwritten == 0
                    && act_ok
                    && refc.decode_ok
                    && refc.fail.is_none()
                    && refc.bits_differ == 0;
                println!(
                    "gemm case={} T={n_tok} routing={r:?} slots={n_slots} unwritten={unwritten} act_codes={} \
                     decode_vs_ggml={} checked={} worst_err_over_band={:.3} contract_bits_differ={} \
                     gemv_max_rel_diff={diag} y_fnv={:016x} {}",
                    st.name,
                    if act_ok { "same" } else { "DIFFER" },
                    if refc.decode_ok { "same" } else { "DIFFER" },
                    refc.checked,
                    refc.worst_ratio,
                    refc.bits_differ,
                    Fnv1a64::default().f32s(&y).value(),
                    verdict(pass)
                );
                if let Some(f) = &refc.fail {
                    println!(
                        "gemm case={} T={n_tok} routing={r:?} first_failure: {f}",
                        st.name
                    );
                }
                if let Some(f) = &refc.bits_fail {
                    println!(
                        "gemm case={} T={n_tok} routing={r:?} first_contract_bits_failure: {f}",
                        st.name
                    );
                }
                ok &= pass;
            }
        }

        // Rerun and graph replay, 64 tokens, uniform routing.
        let n_tok = 64;
        let ids = route_ids(Routing::Uniform, n_tok, top_k, st.n_exp, seed ^ 0xabcd);
        let n_slots = ids.len();
        let x = activations(st.k, st.input.cols(n_slots), seed as u32 ^ 0x5151);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y1 = dev.run(st, &mut res, &xd, &ids)?;
        let y2 = dev.run(st, &mut res, &xd, &ids)?;
        let rerun = bits_equal(&y1, &y2);
        res.y.copy_from_host(stream, &vec![SENT; res.y.len()])?;
        let (gk, w, act, route, ids_d, y) = (
            &dev.gk,
            &res.w,
            &res.act,
            &mut res.route,
            &res.ids,
            &mut res.y,
        );
        let (ty, rows, input) = (st.ty, st.rows, st.input.gemm());
        let sink = dev.gpu.unlabelled_sink();
        let graph = dev.gpu.capture(|s| {
            gk.enqueue_route(s, ids_d, n_slots, route, sink)?;
            gk.enqueue_gemm(
                s,
                GemmArgs {
                    ty,
                    w,
                    rows_per_expert: rows,
                    act,
                    route,
                    input,
                    y,
                },
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let mut yg = res.y.to_host_vec(stream)?;
        yg.truncate(n_slots * st.rows);
        let replay = bits_equal(&yg, &y1);
        let nodes = graph.node_count();
        let pass = rerun && replay && nodes == 2;
        println!(
            "gemm case={} rerun_bits={rerun} graph_nodes={nodes} replay_eq_eager={replay} wall_s={:.1} {}",
            st.name,
            t0.elapsed().as_secs_f64(),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// Write the f16 `d` (`bits`) of one i-quant super-block at the start of
    /// `blk`. IQ1_M has no `d` field: its f16 is spread over the top nibbles
    /// of the four scale halfwords.
    fn put_d(ty: GemmWeight, blk: &mut [u8], bits: u16) {
        if ty == GemmWeight::Iq1M {
            set_d(IqFormat::Iq1M, blk, bits);
        } else {
            blk[..2].copy_from_slice(&bits.to_le_bytes());
        }
    }

    /// Synthetic super-blocks of `ty` with finite f16 scales: every other
    /// byte from the LCG, `d`/`dmin` of exponent 3..=9 (2^-12 .. 2^-5).
    fn synthetic(ty: GemmWeight, n_sb_total: usize, seed: u64) -> Vec<u8> {
        let mut g = Lcg(seed);
        let bpb = ty.block_bytes();
        let mut v = vec![0u8; n_sb_total * bpb];
        let f16 = |g: &mut Lcg| -> [u8; 2] {
            let e = 3 + g.below(7) as u16;
            let bits = (e << 10) | (g.next() as u16 & 0x3ff);
            bits.to_le_bytes()
        };
        for sb in v.chunks_exact_mut(bpb) {
            for b in sb.iter_mut() {
                *b = g.next() as u8;
            }
            let (a, c) = (f16(&mut g), f16(&mut g));
            match ty {
                GemmWeight::Q4K | GemmWeight::Q5K => {
                    sb[0..2].copy_from_slice(&a);
                    sb[2..4].copy_from_slice(&c);
                }
                GemmWeight::Q6K => sb[208..210].copy_from_slice(&a),
                GemmWeight::Q3K => sb[108..110].copy_from_slice(&a),
                GemmWeight::Iq2Xxs
                | GemmWeight::Iq2Xs
                | GemmWeight::Iq2S
                | GemmWeight::Iq1S
                | GemmWeight::Iq1M
                | GemmWeight::Iq3Xxs
                | GemmWeight::Iq3S
                | GemmWeight::Iq4Xs => put_d(ty, sb, u16::from_le_bytes(a)),
            }
        }
        v
    }

    /// The super-blocks of an i-quant stack: `$BLOOMERY_DATA/ref-synth/
    /// <type>.blocks` (`dequant_ref --synthetic`'s dump, `just gate-1-1`)
    /// cycled to `n_sb_total`, or seeded random super-blocks of [`synthetic`]
    /// when the dump is not there.
    fn synth_blocks(ty: GemmWeight, n_sb_total: usize, seed: u64) -> Vec<u8> {
        let name = match ty {
            GemmWeight::Iq3Xxs => "iq3_xxs",
            GemmWeight::Iq4Xs => "iq4_xs",
            _ => return synthetic(ty, n_sb_total, seed),
        };
        let path = data_dir().join("ref-synth").join(format!("{name}.blocks"));
        if let Ok(all) = std::fs::read(&path) {
            let bb = ty.block_bytes();
            if !all.is_empty() && all.len().is_multiple_of(bb) {
                let n = all.len() / bb;
                return (0..n_sb_total)
                    .flat_map(|i| all[bb * (i % n)..][..bb].iter().copied())
                    .collect();
            }
        }
        synthetic(ty, n_sb_total, seed)
    }

    /// The first layer's tensor named by `stem` (`blk.{l}.<stem>`) of type
    /// `ty`, from a table lookup `find`.
    fn first_layer<'a, F>(find: F, stem: &str, ty: GgmlType) -> Option<(usize, &'a [u8], Vec<u64>)>
    where
        F: Fn(&str) -> Option<(GgmlType, &'a [u8], Vec<u64>)>,
    {
        (0..64).find_map(|l| {
            let (t, b, dims) = find(&format!("blk.{l}.{stem}"))?;
            (t == ty).then_some((l, b, dims))
        })
    }

    /// A [`GemmAct`]'s five planes, read back: q3, q4, q6, s8, d8.
    type Planes = (Vec<u64>, Vec<u32>, Vec<u32>, Vec<i32>, Vec<f32>);

    fn planes(act: &GemmAct, stream: &cuda_core::CudaStream) -> Result<Planes, GateError> {
        Ok((
            act.q3().to_host_vec(stream)?,
            act.q4().to_host_vec(stream)?,
            act.q6().to_host_vec(stream)?,
            act.s8().to_host_vec(stream)?,
            act.d8().to_host_vec(stream)?,
        ))
    }

    /// Whether `got` holds block `blk` (its index among the 128-value block
    /// scales) refused: `zero` is what the same input with that block's
    /// values zeroed quantizes to — scale 1.0, zero codes and sums — so a
    /// refused block leaves every plane equal to it but for its scale, NaN.
    fn refused_block(got: &Planes, zero: &Planes, blk: usize) -> bool {
        got.0 == zero.0
            && got.1 == zero.1
            && got.2 == zero.2
            && got.3 == zero.3
            && got.4.len() == zero.4.len()
            && got.4.iter().zip(&zero.4).enumerate().all(|(i, (g, z))| {
                if i == blk {
                    g.is_nan() && *z == 1.0
                } else {
                    g.to_bits() == z.to_bits()
                }
            })
    }

    /// Faults and host refusals (module doc, last paragraph).
    fn faults(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let (k, rows, n_exp, top_k) = (2048usize, 256usize, 16usize, 4usize);
        let st = Stack {
            name: "fault".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x0fa1).into(),
            n_exp,
            rows,
            k,
            input: Input::Shared(top_k),
        };
        let mut res = dev.resident(&st)?;
        let mut ok = true;
        gpu.clear_fault()?;

        // A NaN in token 3's column, in its first 128-value block: the
        // quantizer refuses the block and raises, the named error, and every
        // slot reading the column (its `top_k` slots) is NaN. The same input
        // with that block zeroed is the yardstick: its planes are the refused
        // ones but for the block's scale (1.0 there), and every other slot's
        // outputs are its bits.
        let n_tok = 8;
        let ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 17);
        let mut x = activations(k, n_tok, 991);
        x[3 * k..3 * k + 128].fill(0.0);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y_zero = dev.run(&st, &mut res, &xd, &ids)?;
        let zero = planes(&res.act, stream)?;
        let zero_clean = gpu.take_fault()?.is_none();
        x[3 * k + 77] = f32::NAN;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let fault = gpu.take_fault()?;
        let err = fault.map(|fault| GpuError::fault("gate_gemm", fault));
        let site_ok = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
        let block_refused = refused_block(&planes(&res.act, stream)?, &zero, 3 * (k / 128));
        let reads_col3 = |s: usize| s / top_k == 3;
        let col3_nan = (0..ids.len())
            .filter(|&s| reads_col3(s))
            .all(|s| y[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
        let others_same = (0..ids.len()).filter(|&s| !reads_col3(s)).all(|s| {
            bits_equal(
                &y[s * rows..(s + 1) * rows],
                &y_zero[s * rows..(s + 1) * rows],
            )
        });
        let nan_ok = zero_clean && site_ok && block_refused && col3_nan && others_same;
        println!(
            "gemm fault=nan_activation error=\"{}\" zeroed_block_run_clean={zero_clean} \
             block_refused(d8 NaN, codes and sums zero, other bytes as zeroed)={block_refused} \
             column_3_slots_nan={col3_nan} other_slots_bit_identical={others_same} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        ok &= nan_ok;

        // Ids past the stack: the route raises, leaves the slots out of the
        // tiles and lists them at the table's end, and every GEMM writes
        // their output rows NaN.
        let mut ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 23);
        ids[5] = n_exp as u32;
        ids[9] = u32::MAX;
        let x = activations(k, n_tok, 993);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let err = gpu
            .take_fault()?
            .map(|fault| GpuError::fault("gate_gemm", fault));
        let site_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
            if fault.site() == Some(FaultSite::ExpertId));
        let refused_nan = [5usize, 9]
            .iter()
            .all(|&s| y[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
        let others_written = (0..ids.len()).filter(|s| *s != 5 && *s != 9).all(|s| {
            y[s * rows..(s + 1) * rows]
                .iter()
                .all(|v| v.to_bits() != SENT.to_bits() && !v.is_nan())
        });
        let (cols, tiles) = res.route.read_back(stream)?;
        let listed: usize = tiles.iter().map(|t| t.len as usize).sum();
        let refused = res.route.refused_back(stream)?;
        let table_ok = listed == ids.len() - 2
            && !cols[..listed].contains(&5)
            && !cols[..listed].contains(&9)
            && refused == [5, 9];
        let pass = site_ok && refused_nan && others_written && table_ok;
        println!(
            "gemm fault=expert_id_out_of_range error=\"{}\" refused_slots_nan={refused_nan} \
             other_slots_written={others_written} table_lists={listed}_of_{} refused_listed={refused:?} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            ids.len(),
            verdict(pass)
        );
        ok &= pass;

        // Host refusals: each call must be a named error.
        let unfilled = GemmRoute::new(stream, 64, n_exp)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, 64 * rows)?;
        let mut refusals: Vec<(&str, bool)> = vec![
            ("type_q8_0", GemmWeight::from_ggml(GgmlType::Q8_0).is_err()),
            ("type_q2_k", GemmWeight::from_ggml(GgmlType::Q2_K).is_err()),
            ("k_not_256", GemmAct::new(stream, 8, 1000).is_err()),
            (
                "slots_over_limit",
                GemmRoute::new(stream, GEMM_MAX_SLOTS + 1, n_exp).is_err(),
            ),
            (
                "act_over_limit",
                GemmAct::new(stream, GEMM_MAX_SLOTS + 1, k).is_err(),
            ),
            (
                "unfilled_route",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &unfilled,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "rows_not_16",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: 250,
                            act: &res.act,
                            route: &res.route,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "wrong_type_for_stack",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: GemmWeight::Q6K,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &res.route,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "top_k_not_dividing",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &res.route,
                            input: GemmInput::Shared { top_k: 3 },
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
        ];
        let small = GemmAct::new(stream, 1, k)?;
        refusals.push((
            "act_too_small",
            dev.gk
                .enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty: st.ty,
                        w: &res.w,
                        rows_per_expert: rows,
                        act: &small,
                        route: &res.route,
                        input: st.input.gemm(),
                        y: &mut y,
                    },
                )
                .is_err(),
        ));
        // The Q4_K staging copies the stack in 16-byte pieces: a window one
        // word into an allocation is aligned for u32 and not for that. The
        // refusal comes before any launch, so the window's bytes are never
        // read; the error must name the alignment, not another shape check.
        let pad = DeviceBuffer::<u32>::zeroed(stream, res.w.buf().len() + 4)?;
        let w_off = DeviceTensor::<u32>::window_of(&pad, 4, res.w.rows(), res.w.cols())?;
        let misaligned = dev.gk.enqueue_gemm(
            stream,
            GemmArgs {
                ty: GemmWeight::Q4K,
                w: &w_off,
                rows_per_expert: rows,
                act: &res.act,
                route: &res.route,
                input: st.input.gemm(),
                y: &mut y,
            },
        );
        refusals.push((
            "q4k_stack_misaligned",
            misaligned.is_err_and(|e| e.to_string().contains("16-byte aligned")),
        ));
        let mut dense = GemmRoute::new(stream, 64, 1)?;
        refusals.push((
            "dense_over_capacity",
            dev.gk
                .enqueue_route_dense(stream, 65, &mut dense, gpu.unlabelled_sink())
                .is_err(),
        ));
        refusals.push((
            "dense_on_moe_table",
            dev.gk
                .enqueue_route_dense(stream, 8, &mut res.route, gpu.unlabelled_sink())
                .is_err(),
        ));
        let all = refusals.iter().all(|(_, r)| *r);
        println!(
            "gemm refusals {} {}",
            refusals
                .iter()
                .map(|(n, r)| format!("{n}={}", if *r { "refused" } else { "ACCEPTED" }))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(all)
        );
        ok &= all;
        Ok(ok)
    }

    /// The dense special case: a one-expert Q4_K stack at 2048 × 2048 run
    /// through `enqueue_route_dense` for T ∈ TOKENS, against the reference.
    fn dense_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("dense") {
            return Ok(true);
        }
        let stream = dev.gpu.stream();
        let (k, rows) = (2048usize, 2048usize);
        let st = Stack {
            name: "dense_q4k_2048x2048".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, rows * k / 256, 0xde45e).into(),
            n_exp: 1,
            rows,
            k,
            input: Input::PerSlot(1),
        };
        let n_rows = rows;
        let words = bytes_to_words(&st.bytes);
        let w = DeviceTensor::upload(stream, &words, n_rows, words.len() / n_rows)?;
        let max = TOKENS[TOKENS.len() - 1];
        let mut act = GemmAct::new(stream, max, k)?;
        let mut route = GemmRoute::new(stream, max, 1)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
        let mut ok = true;
        for &t in &TOKENS {
            let x = activations(k, t, 7001 + t as u32);
            let xd = DeviceBuffer::from_host(stream, &x)?;
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            dev.gpu
                .enqueue_quantize_gemm(&xd, t, &mut act, dev.gpu.unlabelled_sink())?;
            dev.gk
                .enqueue_route_dense(stream, t, &mut route, dev.gpu.unlabelled_sink())?;
            dev.gk.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: st.ty,
                    w: &w,
                    rows_per_expert: rows,
                    act: &act,
                    route: &route,
                    input: GemmInput::PerSlot,
                    y: &mut y,
                },
            )?;
            stream.synchronize()?;
            let mut got = y.to_host_vec(stream)?;
            got.truncate(t * rows);
            let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
            let (codes, d8): (Vec<Vec<i8>>, Vec<Vec<f32>>) =
                (0..t).map(|c| q8_1_codes(&x[c * k..(c + 1) * k])).unzip();
            let ids = vec![0u32; t];
            let refc = check_ref(&st, &ids, &codes, &d8, &got)?;
            let (cols, tiles) = route.read_back(stream)?;
            let table_ok = cols.iter().enumerate().all(|(i, &s)| s as usize == i)
                && tiles.len() == t.div_ceil(64);
            let pass = unwritten == 0
                && refc.decode_ok
                && refc.fail.is_none()
                && refc.bits_differ == 0
                && table_ok;
            println!(
                "gemm case={} T={t} tiles={} identity_table={table_ok} unwritten={unwritten} \
                 checked={} worst_err_over_band={:.3} contract_bits_differ={} y_fnv={:016x} {}",
                st.name,
                tiles.len(),
                refc.checked,
                refc.worst_ratio,
                refc.bits_differ,
                Fnv1a64::default().f32s(&got).value(),
                verdict(pass)
            );
            if let Some(f) = &refc.fail {
                println!("gemm case={} T={t} first_failure: {f}", st.name);
            }
            if let Some(f) = &refc.bits_fail {
                println!(
                    "gemm case={} T={t} first_contract_bits_failure: {f}",
                    st.name
                );
            }
            ok &= pass;
        }
        Ok(ok)
    }

    /// Token counts of the route table case: one token, a ragged pass, the
    /// old ubatch, an odd size, the largest ubatch and one token below it.
    const ROUTE_TOKENS: [usize; 6] = [1, 17, 512, 1000, UBATCH - 1, UBATCH];

    /// The route table the host builds: the slots with an id below `n_exp`
    /// grouped by expert, experts ascending, each expert's slots ascending
    /// (the stable grouping), and each expert's run cut into tiles of at
    /// most `GEMM_BN` slots.
    fn route_ref(ids: &[u32], n_exp: usize) -> (Vec<u32>, Vec<GemmTile>) {
        let mut by_exp: Vec<Vec<u32>> = vec![Vec::new(); n_exp];
        for (s, &e) in ids.iter().enumerate() {
            if (e as usize) < n_exp {
                by_exp[e as usize].push(s as u32);
            }
        }
        let mut cols = Vec::with_capacity(ids.len());
        let mut tiles = Vec::new();
        for (e, slots) in by_exp.iter().enumerate() {
            for run in slots.chunks(GEMM_BN) {
                tiles.push(GemmTile {
                    expert: e as u32,
                    start: cols.len() as u32,
                    len: run.len() as u32,
                });
                cols.extend_from_slice(run);
            }
        }
        (cols, tiles)
    }

    /// Whether the table `route` holds is the host's for `ids`: the tiles
    /// equal, and the slot list's first entries (as many as the tiles list)
    /// equal. Prints one line under `label`.
    fn route_matches(
        dev: &Dev<'_>,
        route: &GemmRoute,
        ids: &[u32],
        n_exp: usize,
        label: &str,
    ) -> Result<bool, GateError> {
        let (want_cols, want_tiles) = route_ref(ids, n_exp);
        let (cols, tiles) = route.read_back(dev.gpu.stream())?;
        let tiles_eq = tiles == want_tiles;
        let cols_eq = cols.len() >= want_cols.len() && cols[..want_cols.len()] == want_cols[..];
        let pass = tiles_eq && cols_eq;
        println!(
            "gemm case=route_table {label} slots={} listed={} tiles={} (host {}) \
             tiles_eq_host={tiles_eq} cols_eq_host={cols_eq} {}",
            ids.len(),
            want_cols.len(),
            tiles.len(),
            want_tiles.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The route table against the host's (module doc), its faults at the
    /// largest slot count and its graph replay.
    fn route_table_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("route_table") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let max_tok = ROUTE_TOKENS[ROUTE_TOKENS.len() - 1];
        let mut ok = true;
        gpu.clear_fault()?;
        // Qwen3's and Qwen3.6's routers, each at its model's shape through
        // the instance table: the stacks' experts and a token's slots.
        let router = |rule, experts, top_k| {
            RouterDims::of(MoeShape {
                rule,
                experts,
                top_k,
            })
        };
        let (q3, q35) = (
            router(rules::SOFTMAX_NORM, 128, 8)?,
            router(rules::SOFTMAX_NORM_GATED, 256, 8)?,
        );
        for (stack, n_exp, top_k) in [
            ("qwen3", q3.experts(), q3.slots()),
            ("qwen35", q35.logits(), q35.slots()),
            ("qwen38", 512usize, 10usize),
            ("v41", 384usize, 6usize),
        ] {
            let max_slots = max_tok * top_k;
            let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            for &t in &ROUTE_TOKENS {
                for (ri, &r) in ROUTINGS.iter().enumerate() {
                    let ids = route_ids(r, t, top_k, n_exp, 0x7a61e ^ (t as u64) << 8 ^ ri as u64);
                    ids_d.copy_from_host(stream, &pad(&ids, max_slots))?;
                    dev.gk
                        .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                    let label = format!("stack={stack} T={t} routing={r:?}");
                    ok &= route_matches(dev, &route, &ids, n_exp, &label)?;
                }
            }
            if gpu.take_fault()?.is_some() {
                println!("gemm case=route_table stack={stack}: a fault on valid ids FAIL");
                ok = false;
            }
            if stack != "qwen3" {
                continue;
            }
            // Ids past the stack in three chunks of the largest table: the
            // named fault, the table without their slots, and their slots
            // listed as refused, in slot order.
            let mut ids = route_ids(Routing::Uniform, max_tok, top_k, n_exp, 0xbad1d);
            let bad = [0usize, max_slots / 2 + 1, max_slots - 1];
            ids[bad[0]] = n_exp as u32;
            ids[bad[1]] = u32::MAX;
            ids[bad[2]] = n_exp as u32 + 7;
            ids_d.copy_from_host(stream, &ids)?;
            dev.gk
                .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
            let fault = gpu.take_fault()?;
            let site_ok = fault.is_some_and(|f| f.site() == Some(FaultSite::ExpertId));
            let refused = route.refused_back(stream)?;
            let listed_ok = refused.iter().map(|&s| s as usize).eq(bad);
            println!(
                "gemm case=route_table fault=ids_past_stack slots={max_slots} at {bad:?}: fault=\"{}\" \
                 (want ExpertId) refused_listed={refused:?} {}",
                fault.map_or_else(|| "none".to_string(), |f| f.to_string()),
                verdict(site_ok && listed_ok)
            );
            ok &= site_ok && listed_ok;
            ok &= route_matches(dev, &route, &ids, n_exp, "fault=ids_past_stack")?;
            // A captured route's replay writes the eager table.
            let ids = route_ids(Routing::Uniform, max_tok, top_k, n_exp, 0x9a9f);
            ids_d.copy_from_host(stream, &ids)?;
            let n_slots = ids.len();
            let (gk, route_m, ids_ref) = (&dev.gk, &mut route, &ids_d);
            let graph = gpu.capture(|s| gk.enqueue_route(s, ids_ref, n_slots, route_m, sink))?;
            graph.launch(stream)?;
            stream.synchronize()?;
            ok &= route_matches(dev, &route, &ids, n_exp, "graph_replay")?;
        }
        // The dense table: every slot on expert 0, the identity list.
        let mut dense = GemmRoute::new(stream, max_tok, 1)?;
        for &t in &ROUTE_TOKENS {
            dev.gk.enqueue_route_dense(stream, t, &mut dense, sink)?;
            ok &= route_matches(
                dev,
                &dense,
                &vec![0u32; t],
                1,
                &format!("stack=dense T={t}"),
            )?;
        }
        if gpu.take_fault()?.is_some() {
            println!("gemm case=route_table stack=dense: a fault on the dense table FAIL");
            ok = false;
        }
        Ok(ok)
    }

    /// Column counts of the SwiGLU quantizer case.
    const SWIGLU_COLS: [usize; 4] = [1, 8, 64, 4096];

    /// The SwiGLU quantizer against the two-launch composition, the host
    /// transcription and the f64 SwiGLU (module doc), its fault and its
    /// refusals.
    fn swiglu_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("swiglu") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let mut ok = true;
        for k in [768usize, 2048] {
            let max = SWIGLU_COLS[SWIGLU_COLS.len() - 1];
            let mut fused = GemmAct::new(stream, max, k)?;
            let mut split = GemmAct::new(stream, max, k)?;
            let mut h = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
            for &n in &SWIGLU_COLS {
                let gh = activations(k, n, 4400 + n as u32 + k as u32);
                let uh = activations(k, n, 5500 + n as u32 + k as u32);
                let (g, u) = (
                    DeviceBuffer::from_host(stream, &gh)?,
                    DeviceBuffer::from_host(stream, &uh)?,
                );
                dev.gk.enqueue_swiglu_quant(
                    stream,
                    &g,
                    &u,
                    n,
                    &mut fused,
                    gpu.unlabelled_sink(),
                )?;
                gpu.elem().enqueue_swiglu(stream, &g, &u, n * k, &mut h)?;
                gpu.enqueue_quantize_gemm(&h, n, &mut split, gpu.unlabelled_sink())?;
                stream.synchronize()?;
                let planes_same = fused.q3().to_host_vec(stream)?
                    == split.q3().to_host_vec(stream)?
                    && fused.q4().to_host_vec(stream)? == split.q4().to_host_vec(stream)?
                    && fused.q6().to_host_vec(stream)? == split.q6().to_host_vec(stream)?
                    && fused.s8().to_host_vec(stream)? == split.s8().to_host_vec(stream)?
                    && bits_equal(
                        &fused.d8().to_host_vec(stream)?,
                        &split.d8().to_host_vec(stream)?,
                    );
                let hd = h.to_host_vec(stream)?;
                let (dev_d8, dev_s8) = (
                    fused.d8().to_host_vec(stream)?,
                    fused.s8().to_host_vec(stream)?,
                );
                let nb = k / 128;
                let host_ok = (0..n).all(|c| {
                    let (codes, d8) = q8_1_codes(&hd[c * k..(c + 1) * k]);
                    (0..nb).all(|b| dev_d8[c * nb + b].to_bits() == d8[b].to_bits())
                        && (0..k / 32).all(|j| {
                            let s: i32 = codes[32 * j..32 * j + 32]
                                .iter()
                                .map(|&v| i32::from(v))
                                .sum();
                            dev_s8[c * (k / 32) + j] == s
                        })
                });
                let mut worst = 0.0f64;
                let mut silu_ok = true;
                for i in 0..n * k {
                    let (gv, uv) = (f64::from(gh[i]), f64::from(uh[i]));
                    let want = gv / (1.0 + (-gv).exp()) * uv;
                    let tol =
                        4.0 * f64::from(f32::EPSILON) * want.abs() + f64::from(f32::MIN_POSITIVE);
                    let err = (f64::from(hd[i]) - want).abs();
                    silu_ok &= err <= tol;
                    worst = worst.max(err / tol);
                }
                let pass = planes_same && host_ok && silu_ok;
                println!(
                    "gemm case=swiglu K={k} cols={n} planes_eq_split_bits={planes_same} \
                     host_scales_and_sums={host_ok} silu_within_4ulp={silu_ok} \
                     worst_err_over_tol={worst:.3} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        // A NaN in one gate row, in block 2 of column 5: the named fault and
        // that block refused, against the same rows with the block's gate
        // and up values zeroed (a SwiGLU of 0).
        let k = 768;
        let mut act = GemmAct::new(stream, 8, k)?;
        let mut gh = activations(k, 8, 77);
        let mut uh = activations(k, 8, 78);
        gh[5 * k + 256..5 * k + 384].fill(0.0);
        uh[5 * k + 256..5 * k + 384].fill(0.0);
        gpu.clear_fault()?;
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        dev.gk
            .enqueue_swiglu_quant(stream, &g, &u, 8, &mut act, gpu.unlabelled_sink())?;
        let zero = planes(&act, stream)?;
        let zero_clean = gpu.take_fault()?.is_none();
        gh[5 * k + 300] = f32::NAN;
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        dev.gk
            .enqueue_swiglu_quant(stream, &g, &u, 8, &mut act, gpu.unlabelled_sink())?;
        let fault = gpu.take_fault()?;
        let err = fault.map(|fault| GpuError::fault("gate_gemm", fault));
        let block_refused = refused_block(&planes(&act, stream)?, &zero, 5 * (k / 128) + 2);
        let nan_ok = zero_clean
            && fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn))
            && block_refused;
        println!(
            "gemm case=swiglu fault=nan_gate error=\"{}\" zeroed_block_run_clean={zero_clean} \
             block_refused(d8 NaN, codes and sums zero, other bytes as zeroed)={block_refused} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        let short = DeviceBuffer::<f32>::zeroed(stream, k)?;
        let sink = gpu.unlabelled_sink();
        let refusals = [
            (
                "no_columns",
                dev.gk
                    .enqueue_swiglu_quant(stream, &g, &u, 0, &mut act, sink)
                    .is_err(),
            ),
            (
                "past_act",
                dev.gk
                    .enqueue_swiglu_quant(stream, &g, &u, 9, &mut act, sink)
                    .is_err(),
            ),
            (
                "short_input",
                dev.gk
                    .enqueue_swiglu_quant(stream, &short, &u, 2, &mut act, sink)
                    .is_err(),
            ),
        ];
        let all = refusals.iter().all(|(_, r)| *r);
        println!(
            "gemm case=swiglu refusals {} {}",
            refusals
                .iter()
                .map(|(n, r)| format!("{n}={}", if *r { "refused" } else { "ACCEPTED" }))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(all)
        );
        Ok(ok && nan_ok && all)
    }

    /// Clamp limits of the clamped SwiGLU quantizer case: GLM-5.3's 10, and
    /// 0.5, under the activations' scale, so both clamps (`silu(g)` above the
    /// limit, `u` outside ±limit) act on a share of the values.
    const SWIGLU_LIMITS: [f32; 2] = [10.0, 0.5];
    /// The limits the OAI rule runs at: the references' 7, and 0.5, which bites on most values.
    const SWIGLU_OAI_LIMITS: [f32; 2] = [7.0, 0.5];

    /// The squared-ReLU quantizer fold (`Gpu::enqueue_quantize_gemm_relu_sqr`)
    /// against the two-launch composition and the host transcription
    /// (module doc), its fault.
    fn relu_sqr_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("relu_sqr") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let mut ok = true;
        for k in [768usize, 2048] {
            let max = SWIGLU_COLS[SWIGLU_COLS.len() - 1];
            let mut fused = GemmAct::new(stream, max, k)?;
            let mut split = GemmAct::new(stream, max, k)?;
            let mut h = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
            for &n in &SWIGLU_COLS {
                let xh = activations(k, n, 8800 + n as u32 + k as u32);
                let x = DeviceBuffer::from_host(stream, &xh)?;
                gpu.enqueue_quantize_gemm_relu_sqr(&x, n, &mut fused, gpu.unlabelled_sink())?;
                gpu.elem().enqueue_relu_sqr(stream, &x, n * k, &mut h)?;
                gpu.enqueue_quantize_gemm(&h, n, &mut split, gpu.unlabelled_sink())?;
                stream.synchronize()?;
                let planes_same = fused.q3().to_host_vec(stream)?
                    == split.q3().to_host_vec(stream)?
                    && fused.q4().to_host_vec(stream)? == split.q4().to_host_vec(stream)?
                    && fused.q6().to_host_vec(stream)? == split.q6().to_host_vec(stream)?
                    && fused.s8().to_host_vec(stream)? == split.s8().to_host_vec(stream)?
                    && bits_equal(
                        &fused.d8().to_host_vec(stream)?,
                        &split.d8().to_host_vec(stream)?,
                    );
                let th: Vec<f32> = xh
                    .iter()
                    .map(|&v| {
                        let r = if v > 0.0 { v } else { 0.0 };
                        r * r
                    })
                    .collect();
                let (dev_d8, dev_s8) = (
                    fused.d8().to_host_vec(stream)?,
                    fused.s8().to_host_vec(stream)?,
                );
                let nb = k / 128;
                let host_ok = (0..n).all(|c| {
                    let (codes, d8) = q8_1_codes(&th[c * k..(c + 1) * k]);
                    (0..nb).all(|b| dev_d8[c * nb + b].to_bits() == d8[b].to_bits())
                        && (0..k / 32).all(|j| {
                            let s: i32 = codes[32 * j..32 * j + 32]
                                .iter()
                                .map(|&v| i32::from(v))
                                .sum();
                            dev_s8[c * (k / 32) + j] == s
                        })
                });
                let transform_ok = bits_equal(&h.to_host_vec(stream)?[..n * k], &th[..n * k]);
                let pass = planes_same && host_ok && transform_ok;
                println!(
                    "gemm case=relu_sqr K={k} cols={n} planes_eq_split_bits={planes_same} \
                     host_scales_and_sums={host_ok} transform_bits={transform_ok} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        // A NaN in one row, in block 2 of column 5: the named fault and that
        // block refused, against the same rows with the block's values
        // zeroed (a squared ReLU of 0).
        let k = 768;
        let mut act = GemmAct::new(stream, 8, k)?;
        let mut xh = activations(k, 8, 88);
        xh[5 * k + 256..5 * k + 384].fill(0.0);
        gpu.clear_fault()?;
        let x = DeviceBuffer::from_host(stream, &xh)?;
        gpu.enqueue_quantize_gemm_relu_sqr(&x, 8, &mut act, gpu.unlabelled_sink())?;
        let zero = planes(&act, stream)?;
        let zero_clean = gpu.take_fault()?.is_none();
        xh[5 * k + 300] = f32::NAN;
        let x = DeviceBuffer::from_host(stream, &xh)?;
        gpu.enqueue_quantize_gemm_relu_sqr(&x, 8, &mut act, gpu.unlabelled_sink())?;
        let fault = gpu.take_fault()?;
        let err = fault.map(|fault| GpuError::fault("gate_gemm", fault));
        let block_refused = refused_block(&planes(&act, stream)?, &zero, 5 * (k / 128) + 2);
        let nan_ok = zero_clean
            && fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn))
            && block_refused;
        println!(
            "gemm case=relu_sqr fault=nan_input error=\"{}\" zeroed_block_run_clean={zero_clean} \
             block_refused(d8 NaN, codes and sums zero, other bytes as zeroed)={block_refused} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        Ok(ok && nan_ok)
    }

    /// The clamped SwiGLU quantizer (`GemmKernels::enqueue_swiglu_act_quant`
    /// under `Act::SwigluClamp`) against the CPU tier's ik-verified
    /// `qdot::swiglu_clamp` quantized by `Gpu::enqueue_quantize_gemm`: the
    /// same planes bit for bit, and a `Q8Act` filled by
    /// `enqueue_swiglu_act_quant_q8` holding the `GemmAct`'s codes and
    /// scales. At limit 0.5 values reach the clamp and the planes differ from
    /// the unclamped rule's. A NaN up value is refused as `QuantColumn` where the
    /// clamp alone would carry it to the limit, and a non-finite limit is
    /// refused by name.
    fn swiglu_clamp_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("swiglu") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let mut ok = true;
        for k in [768usize, 2048] {
            let n = 8;
            let scale = |v: Vec<f32>| v.into_iter().map(|x| 2.0 * x).collect::<Vec<f32>>();
            let gh = scale(activations(k, n, 6600 + k as u32));
            let uh = scale(activations(k, n, 6700 + k as u32));
            let (g, u) = (
                DeviceBuffer::from_host(stream, &gh)?,
                DeviceBuffer::from_host(stream, &uh)?,
            );
            let mut plain = GemmAct::new(stream, n, k)?;
            dev.gk
                .enqueue_swiglu_quant(stream, &g, &u, n, &mut plain, sink)?;
            let rules = SWIGLU_LIMITS
                .map(|limit| (Act::SwigluClamp { limit }, limit))
                .into_iter()
                .chain(SWIGLU_OAI_LIMITS.map(|limit| (swiglu_host::oai(limit), limit)));
            for (rule, limit) in rules {
                let kind = if matches!(rule, Act::SwigluOai { .. }) {
                    "swiglu_oai"
                } else {
                    "swiglu_clamp"
                };
                let (hh, live) = swiglu_host::combine(rule, &gh, &uh).expect("a limited rule");
                let h = DeviceBuffer::from_host(stream, &hh)?;
                let mut fused = GemmAct::new(stream, n, k)?;
                let mut split = GemmAct::new(stream, n, k)?;
                let mut gemv = Q8Act::with_k(stream, n, k)?;
                dev.gk
                    .enqueue_swiglu_act_quant(stream, &g, &u, rule, n, &mut fused, sink)?;
                gpu.enqueue_quantize_gemm(&h, n, &mut split, sink)?;
                dev.gk
                    .enqueue_swiglu_act_quant_q8(stream, &g, &u, rule, n, &mut gemv, sink)?;
                stream.synchronize()?;
                let (pf, ps, pp) = (
                    planes(&fused, stream)?,
                    planes(&split, stream)?,
                    planes(&plain, stream)?,
                );
                let eq_host_clamp = pf.0 == ps.0
                    && pf.1 == ps.1
                    && pf.2 == ps.2
                    && pf.3 == ps.3
                    && bits_equal(&pf.4, &ps.4);
                let clamp_acts = limit > 1.0 || (live > 0 && pf.0 != pp.0);
                let q8_eq = gemv.q3().to_host_vec(stream)? == pf.0
                    && bits_equal(&gemv.d8().to_host_vec(stream)?, &pf.4);
                let pass = eq_host_clamp && clamp_acts && q8_eq;
                println!(
                    "gemm case={kind} K={k} cols={n} limit={limit} live_values={live} \
                     planes_eq_host_clamp_bits={eq_host_clamp} differs_from_unclamped={clamp_acts} \
                     q8act_eq_gemmact={q8_eq} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        // A NaN up value under the clamp: the clamp alone would make it the
        // limit; the block is refused and the fault named.
        let k = 768;
        let gh = activations(k, 8, 77);
        let mut uh = activations(k, 8, 78);
        uh[3 * k + 5] = f32::NAN;
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        let mut act = GemmAct::new(stream, 8, k)?;
        for (kind, rule) in [
            ("swiglu_clamp", Act::SwigluClamp { limit: 0.5 }),
            ("swiglu_oai", swiglu_host::oai(0.5)),
        ] {
            gpu.clear_fault()?;
            dev.gk
                .enqueue_swiglu_act_quant(stream, &g, &u, rule, 8, &mut act, sink)?;
            stream.synchronize()?;
            let fault = gpu.take_fault()?;
            let refused = act.d8().to_host_vec(stream)?[3 * (k / 128)].is_nan();
            let nan_ok = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn)) && refused;
            println!(
                "gemm case={kind} fault=nan_up fault_raised={} block_refused={refused} {}",
                fault.is_some(),
                verdict(nan_ok)
            );
            ok &= nan_ok;
        }
        // A rule the device cannot run is refused by name, before any launch.
        for (what, rule) in [
            (
                "clamp_inf_limit",
                Act::SwigluClamp {
                    limit: f32::INFINITY,
                },
            ),
            (
                "oai_nan_limit",
                Act::SwigluOai {
                    alpha: SWIGLU_OAI_ALPHA,
                    limit: f32::NAN,
                },
            ),
            ("oai_negative_limit", swiglu_host::oai(-1.0)),
            (
                "oai_alpha_1",
                Act::SwigluOai {
                    alpha: 1.0,
                    limit: 7.0,
                },
            ),
        ] {
            let refused = dev
                .gk
                .enqueue_swiglu_act_quant(stream, &g, &u, rule, 8, &mut act, sink)
                .is_err();
            println!(
                "gemm case=swiglu refusal={what} refused={refused} {}",
                verdict(refused)
            );
            ok &= refused;
        }
        Ok(ok)
    }

    /// The card-slots-only K-quant SwiGLU quantizer
    /// (`GemmKernels::enqueue_swiglu_act_quant_sel`) over a remapped route's
    /// places, under the clamp GLM's routed experts run (10) and the plain
    /// rule: a first plain launch seeds every column's bytes, then the
    /// `_sel` launch over other rows — the card columns the plain launch's
    /// bits over those rows, the host columns the seed's, untouched — a NaN
    /// in a host column raising nothing, a place in `[n_card, HOST)`
    /// `ExpertId` with nothing written. The clause turns red on a mutant
    /// that drops the `HOST` early return (a host column raising), takes the
    /// stray bound above `n_card` (`place > n_card`: a stray column written,
    /// no fault) or quantizes the host columns (the seed's bytes gone).
    fn swiglu_sel_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("swiglu_sel") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let mut ok = true;
        gpu.clear_fault()?;
        let (k, n, n_card) = (768usize, 33usize, 20usize);
        let place = |c: usize| {
            if c.is_multiple_of(3) {
                HOST
            } else {
                ((c * 7) % n_card) as u32
            }
        };
        let sel: Vec<u32> = (0..n).map(place).collect();
        let sel_d = DeviceBuffer::from_host(stream, &sel)?;
        for (name, rule) in [
            ("clamp10", Act::SwigluClamp { limit: 10.0 }),
            ("silu_mul", Act::SiluMul),
        ] {
            let mut plain = GemmAct::new(stream, n, k)?;
            let mut sela = GemmAct::new(stream, n, k)?;
            // Twelve so silu(g) passes the clamp 10 on a share of the values.
            let scaled = |seed: u32| -> Vec<f32> {
                activations(k, n, seed)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect()
            };
            let (g0, u0, g1, u1) = (
                DeviceBuffer::from_host(stream, &scaled(9500))?,
                DeviceBuffer::from_host(stream, &scaled(9501))?,
                DeviceBuffer::from_host(stream, &scaled(9502))?,
                DeviceBuffer::from_host(stream, &scaled(9503))?,
            );
            dev.gk
                .enqueue_swiglu_act_quant(stream, &g0, &u0, rule, n, &mut sela, sink)?;
            stream.synchronize()?;
            let before = planes(&sela, stream)?;
            dev.gk.enqueue_swiglu_act_quant_sel(
                stream, &g1, &u1, rule, &sel_d, n_card, n, &mut sela, sink,
            )?;
            dev.gk
                .enqueue_swiglu_act_quant(stream, &g1, &u1, rule, n, &mut plain, sink)?;
            stream.synchronize()?;
            let fault = gpu.take_fault()?;
            let (got, want) = (planes(&sela, stream)?, planes(&plain, stream)?);
            let (n_sb, half_it, quad_it) = (k / 256, (k / 256).div_ceil(2), (k / 256).div_ceil(4));
            let eq = |a: &Planes, b: &Planes, c: usize| {
                a.0[c * 64 * half_it..][..64 * half_it] == b.0[c * 64 * half_it..][..64 * half_it]
                    && a.1[c * 256 * quad_it..][..256 * quad_it]
                        == b.1[c * 256 * quad_it..][..256 * quad_it]
                    && a.2[c * 128 * half_it..][..128 * half_it]
                        == b.2[c * 128 * half_it..][..128 * half_it]
                    && a.3[c * 8 * n_sb..][..8 * n_sb] == b.3[c * 8 * n_sb..][..8 * n_sb]
                    && bits_equal(
                        &a.4[c * 2 * n_sb..][..2 * n_sb],
                        &b.4[c * 2 * n_sb..][..2 * n_sb],
                    )
            };
            let mut card = 0usize;
            let mut host_ok = true;
            let mut card_ok = true;
            for (c, &sel_c) in sel.iter().enumerate().take(n) {
                if sel_c == HOST {
                    host_ok &= eq(&got, &before, c);
                } else {
                    card += 1;
                    card_ok &= eq(&got, &want, c);
                }
            }
            let pass = card_ok && host_ok && fault.is_none();
            println!(
                "gemm case=swiglu_sel rule={name} K={k} cols={n} card_cols_eq_plain={card_ok} \
                 ({card} of them) host_cols_untouched={host_ok} fault={} {}",
                fault.map_or_else(|| "none".into(), |f| f.to_string()),
                verdict(pass)
            );
            ok &= pass;
            // A NaN in a host column is never read: no fault, the host
            // column still the seed's.
            let mut gn = scaled(9504);
            let host_col = (0..n).find(|&c| sel[c] == HOST).unwrap_or(0);
            gn[host_col * k + 5] = f32::NAN;
            let gn = DeviceBuffer::from_host(stream, &gn)?;
            dev.gk.enqueue_swiglu_act_quant_sel(
                stream, &gn, &u1, rule, &sel_d, n_card, n, &mut sela, sink,
            )?;
            stream.synchronize()?;
            let fault = gpu.take_fault()?;
            let got = planes(&sela, stream)?;
            let host_kept = eq(&got, &before, host_col);
            let pass = fault.is_none() && host_kept;
            println!(
                "gemm case=swiglu_sel rule={name} fault=nan_host col={host_col} \
                 host_col_untouched={host_kept} fault_raised={} {}",
                fault.is_some(),
                verdict(pass)
            );
            ok &= pass;
            // A place in [n_card, HOST): `ExpertId`, its column unwritten.
            let mut stray = sel.clone();
            stray[host_col] = n_card as u32;
            let stray_d = DeviceBuffer::from_host(stream, &stray)?;
            dev.gk.enqueue_swiglu_act_quant_sel(
                stream, &g1, &u1, rule, &stray_d, n_card, n, &mut sela, sink,
            )?;
            stream.synchronize()?;
            let fault = gpu.take_fault()?;
            let got = planes(&sela, stream)?;
            let unwritten = eq(&got, &before, host_col);
            let pass = fault == Some(Fault::at(LAYER_NONE, FaultSite::ExpertId)) && unwritten;
            println!(
                "gemm case=swiglu_sel rule={name} fault=stray_place col={host_col} \
                 col_unwritten={unwritten} fault={} {}",
                fault.map_or_else(|| "none".into(), |f| f.to_string()),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        if std::env::args().any(|a| a == "--bench-kernels") {
            return bench::run(&gpu);
        }
        println!("gemm device={}", gpu.device_name()?);
        let dev = Dev {
            gpu: &gpu,
            gk: GemmKernels::load(gpu.context())?,
            q6s: Q6kSelKernels::load(gpu.context(), gpu.fault_word())?,
            iqk: IqKernels::load(gpu.context())?,
        };
        let mut ok = true;

        if matches!(
            case_filter().as_deref(),
            Some(
                "dense_mma"
                    | "bf16_store"
                    | "g32_slab"
                    | "g32_routed"
                    | "g38_iq4nl"
                    | "q5_0p_is_the_gemv"
                    | "norm_form"
                    | "mxfp8_w8a8"
                    | "nvfp4_native"
                    | "nvfp4_swiglu"
                    | "swiglu_epi"
                    | "nvfp4_gate_up"
                    | "zgate"
            )
        ) {
            return if g32::run(&gpu, &dev.gk)? {
                Ok(())
            } else {
                Err(checks_failed())
            };
        }

        // Qwen3: the model file of this profile.
        let qwen = open_model()?;
        let find_q = |name: &str| {
            let t = qwen.find(name)?;
            Some((t.ty, qwen.data(t).ok()?, t.dims.clone()))
        };
        let (lg, gate_b, gdims) = first_layer(find_q, "ffn_gate_exps.weight", GgmlType::Q4_K)
            .ok_or("no Q4_K ffn_gate_exps in the qwen3moe file")?;
        let (ld4, down4_b, d4dims) = first_layer(find_q, "ffn_down_exps.weight", GgmlType::Q4_K)
            .ok_or("no Q4_K ffn_down_exps in the qwen3moe file")?;
        let (ld6, down6_b, d6dims) = first_layer(find_q, "ffn_down_exps.weight", GgmlType::Q6_K)
            .ok_or("no Q6_K ffn_down_exps in the qwen3moe file")?;
        println!(
            "gemm qwen3 gate=blk.{lg} {gdims:?} down_q4k=blk.{ld4} {d4dims:?} down_q6k=blk.{ld6} {d6dims:?}"
        );
        let dims = |d: &[u64]| (d[0] as usize, d[1] as usize, d[2] as usize);
        let (gk_, gr, ge) = dims(&gdims);
        let (dk, dr, de) = dims(&d4dims);
        let cases_q = [
            (
                "qwen3_gate_q4k",
                GemmWeight::Q4K,
                gate_b,
                gk_,
                gr,
                ge,
                Input::Shared(8),
            ),
            (
                "qwen3_down_q4k",
                GemmWeight::Q4K,
                down4_b,
                dk,
                dr,
                de,
                Input::PerSlot(8),
            ),
            (
                "qwen3_down_q6k",
                GemmWeight::Q6K,
                down6_b,
                dk,
                dr,
                de,
                Input::PerSlot(8),
            ),
        ];
        for (i, (name, ty, b, k, rows, n_exp, input)) in cases_q.into_iter().enumerate() {
            let st = Stack {
                name: format!("{name}_real"),
                ty,
                bytes: b.into(),
                n_exp,
                rows,
                k,
                input,
            };
            ok &= run_case(&dev, &st, 0x51 + i as u64)?;
            if !wanted(&format!("{name}_synth")) {
                continue;
            }
            let syn = synthetic(ty, n_exp * rows * k / 256, 0x5e0 + i as u64);
            let st = Stack {
                name: format!("{name}_synth"),
                ty,
                bytes: syn.into(),
                n_exp,
                rows,
                k,
                input,
            };
            ok &= run_case(&dev, &st, 0x61 + i as u64)?;
        }
        if wanted("qwen3_gate_shape_q5k_synth") {
            let syn = synthetic(GemmWeight::Q5K, ge * gr * gk_ / 256, 0x5e5);
            let st = Stack {
                name: "qwen3_gate_shape_q5k_synth".into(),
                ty: GemmWeight::Q5K,
                bytes: syn.into(),
                n_exp: ge,
                rows: gr,
                k: gk_,
                input: Input::Shared(8),
            };
            ok &= run_case(&dev, &st, 0x71)?;
        }
        if wanted("q4k_ragged_rows_synth") {
            // 400 rows an expert: three full 128-row slabs and one of 16, whose
            // block has one warp over live rows and seven past them.
            let (n_exp, rows, k) = (16usize, 400usize, 2048usize);
            let syn = synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x5e6);
            let st = Stack {
                name: "q4k_ragged_rows_synth".into(),
                ty: GemmWeight::Q4K,
                bytes: syn.into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case(&dev, &st, 0x72)?;
        }
        drop(qwen);

        if wanted("v41_gate_q3k") {
            ok &= v41_cases(&dev, dims)?;
        }
        ok &= q38_cases(&dev)?;
        ok &= iq2xxs_cases(&dev)?;
        ok &= iq3s_cases(&dev)?;
        ok &= iq_card_cases(&dev)?;
        ok &= dense_case(&dev)?;
        ok &= route_table_case(&dev)?;
        ok &= swiglu_case(&dev)?;
        ok &= relu_sqr_case(&dev)?;
        ok &= swiglu_clamp_case(&dev)?;
        ok &= swiglu_sel_case(&dev)?;
        ok &= faults(&dev)?;
        ok &= sel_faults(&dev)?;
        ok &= site_mask(&dev)?;
        ok &= g32::run(&gpu, &dev.gk)?;

        if !ok {
            return Err(checks_failed());
        }
        if let Some(f) = case_filter() {
            println!("gemm filter --case {f}: only the matching cases ran");
        }
        println!(
            "PASSED: gate_gemm — every run inside its derived band of the f64 reference and bit for bit \
             the contract's transcription, every slot written, rerun and graph replay bit-identical, \
             faults named; the route table the host's stable grouping bit for bit up to 32,768 slots; \
             the SwiGLU quantizer bit for bit its composition; the 32-value family (Q8_0 planes and \
             file, Q5_1, packed Q5_0) bit for bit its contract, packed Q5_0 inside the gemv's \
             band, its quantizers the host's, the remapped route's host slots untouched, the wide \
             F32 product f32_gemv's bits"
        );
        Ok(())
    }

    /// The V4.1 cases: one routed gate stack of the file, and a synthetic
    /// stack of its shape.
    fn v41_cases(
        dev: &Dev<'_>,
        dims: impl Fn(&[u64]) -> (usize, usize, usize),
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let v41 = gguf::Split::open(gguf::v41::model())?;
        let find_v = |name: &str| {
            let (sh, t) = v41.find(name)?;
            Some((t.ty, v41.shard(sh)?.data(t).ok()?, t.dims.clone()))
        };
        let (lv, vb, vdims) = first_layer(find_v, "ffn_gate_exps.weight", GgmlType::Q3_K)
            .ok_or("no Q3_K ffn_gate_exps in the V4.1 file")?;
        println!("gemm v41 gate=blk.{lv} {vdims:?}");
        let (vk, vr, ve) = dims(&vdims);
        let st = Stack {
            name: "v41_gate_q3k_real".into(),
            ty: GemmWeight::Q3K,
            bytes: vb.into(),
            n_exp: ve,
            rows: vr,
            k: vk,
            input: Input::Shared(6),
        };
        ok &= run_case(dev, &st, 0x81)?;
        if wanted("v41_gate_q3k_synth") {
            let syn = synthetic(GemmWeight::Q3K, ve * vr * vk / 256, 0x5e3);
            let st = Stack {
                name: "v41_gate_q3k_synth".into(),
                ty: GemmWeight::Q3K,
                bytes: syn.into(),
                n_exp: ve,
                rows: vr,
                k: vk,
                input: Input::Shared(6),
            };
            ok &= run_case(dev, &st, 0x91)?;
        }
        Ok(ok)
    }

    /// The q8_1 quantizer's Q4_K slot of value-order word `v` (gate_iq's
    /// helper): the i-quant row kernels read their columns in that
    /// permutation.
    fn q4_slot(v: usize) -> usize {
        256 * (v >> 8) + 32 * (v & 7) + 8 * ((v >> 6) & 3) + ((v >> 3) & 7)
    }

    /// The max relative difference of the grouped GEMM's T = 1 outputs to the
    /// i-quant row cores (`iq3_xxs_rows`, `iq4_xs_rows`) over the same host
    /// q8_1 codes: the row kernels run the whole repacked stack
    /// (`IqFormat::repack`) against the run's columns, each slot read at its
    /// expert's rows of the launch's `y`. A diagnostic, not a pin.
    fn iq_rows_diag(
        dev: &Dev<'_>,
        st: &Stack<'_>,
        res: &mut Resident,
        seed: u64,
    ) -> Result<f64, GateError> {
        let stream = dev.gpu.stream();
        let fmt = match st.ty {
            GemmWeight::Iq2Xxs => IqFormat::Iq2Xxs,
            GemmWeight::Iq3Xxs => IqFormat::Iq3Xxs,
            GemmWeight::Iq4Xs => IqFormat::Iq4Xs,
            _ => return Err("iq_rows_diag: not an i-quant stack".into()),
        };
        let ids = route_ids(
            Routing::Uniform,
            1,
            st.input.top_k(),
            st.n_exp,
            seed ^ 0xd1a6,
        );
        let n_slots = ids.len();
        let x = activations(st.k, st.input.cols(n_slots), (seed as u32) ^ 0xd1a7);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(st, res, &xd, &ids)?;
        let rows = IqRows::upload(stream, fmt, &st.bytes, st.n_exp * st.rows, st.k)?;
        let total = st.n_exp * st.rows;
        let (codes, d8) = q8_1_codes(&x[..st.k]);
        let col_words = 256 * (st.k / 256).div_ceil(4);
        let mut out = vec![0.0f32; n_slots * st.rows];
        let mut ybuf = DeviceBuffer::<f32>::zeroed(stream, 8 * total)?;
        for s0 in (0..n_slots).step_by(8) {
            let m = (n_slots - s0).min(8);
            // Every slot of the one token reads its one column.
            let mut q = vec![0u32; m * col_words];
            for c in 0..m {
                for v in 0..st.k / 4 {
                    q[c * col_words + q4_slot(v)] =
                        u32::from_le_bytes(std::array::from_fn(|i| codes[4 * v + i] as u8));
                }
            }
            let (qd, dd) = (
                DeviceBuffer::from_host(stream, &q)?,
                DeviceBuffer::from_host(stream, &d8.repeat(m))?,
            );
            dev.iqk
                .enqueue_rows(stream, &rows, &qd, &dd, m, &mut ybuf)?;
            let got = ybuf.to_host_vec(stream)?;
            for s in s0..s0 + m {
                let at = (s - s0) * total + ids[s] as usize * st.rows;
                out[s * st.rows..(s + 1) * st.rows].copy_from_slice(&got[at..at + st.rows]);
            }
        }
        Ok(rel_diff(&y, &out))
    }

    /// A NaN f16 `d` in the first super-block of every row of expert 0 makes
    /// every output of the slots that read it NaN — no silent failure: a
    /// T = 1 run with every slot on expert 0.
    fn iq_nan_d(dev: &Dev<'_>, st: &Stack<'_>, seed: u64) -> Result<bool, GateError> {
        let stream = dev.gpu.stream();
        let top_k = st.input.top_k();
        let bb = st.ty.block_bytes() * st.k / 256;
        let mut bytes = st.bytes.to_vec();
        for r in 0..st.rows {
            put_d(st.ty, &mut bytes[r * bb..], 0x7e00);
        }
        let st = Stack {
            name: format!("{}_nan_d", st.name),
            ty: st.ty,
            bytes: bytes.into(),
            n_exp: st.n_exp,
            rows: st.rows,
            k: st.k,
            input: st.input,
        };
        let mut res = dev.resident_at(&st, top_k)?;
        let ids = vec![0u32; top_k];
        let x = activations(st.k, st.input.cols(top_k), (seed as u32) ^ 0xd1a9);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let nan = y.iter().all(|v| v.is_nan());
        println!(
            "gemm case={}_nan_d ty={:?} every_output_nan={nan} {}",
            st.name,
            st.ty,
            verdict(nan)
        );
        Ok(nan)
    }

    /// The Qwen3.8 i-quant cases (the card route `arch/qwen3moe/wide38.rs`
    /// wires): synthetic IQ3_XXS and IQ4_XS stacks of the routed gate shape —
    /// 64 experts of 640 rows, K 2560, top-10, every slot its token's column,
    /// the blocks cycled from the ref-synth dump — and an IQ3_XXS stack of
    /// 400 rows an expert (a partial 128-row slab). Each runs the four
    /// routings at [`TOKENS`] through [`run_case`], the T = 1 outputs
    /// against the i-quant row cores, a NaN `d` super-block's outputs NaN,
    /// `from_ggml`'s acceptance of the two types, and the host API's refusal
    /// of the other i-quant type's stack.
    fn q38_cases(dev: &Dev<'_>) -> Result<bool, GateError> {
        if ![
            "q38_iq3xxs_gate_synth",
            "q38_iq4xs_gate_synth",
            "q38_iq3xxs_ragged_rows_synth",
        ]
        .iter()
        .any(|&n| wanted(n))
        {
            return Ok(true);
        }
        let accept = GemmWeight::from_ggml(GgmlType::IQ3_XXS)
            .is_ok_and(|t| t == GemmWeight::Iq3Xxs)
            && GemmWeight::from_ggml(GgmlType::IQ4_XS).is_ok_and(|t| t == GemmWeight::Iq4Xs);
        println!(
            "gemm case=q38_types from_ggml(iq3_xxs, iq4_xs)={} {}",
            if accept { "accepted" } else { "REFUSED" },
            verdict(accept)
        );
        let mut ok = accept;
        for (name, ty, seed) in [
            ("q38_iq3xxs_gate_synth", GemmWeight::Iq3Xxs, 0x5e7u64),
            ("q38_iq4xs_gate_synth", GemmWeight::Iq4Xs, 0x5e9),
        ] {
            if !wanted(name) {
                continue;
            }
            let (n_exp, rows, k, top_k) = (64usize, 640usize, 2560usize, 10usize);
            let st = Stack {
                name: name.into(),
                ty,
                bytes: synth_blocks(ty, n_exp * rows * k / 256, seed).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(top_k),
            };
            let mut res = dev.resident(&st)?;
            ok &= run_case(dev, &st, seed ^ 0x61)?;
            let diag = iq_rows_diag(dev, &st, &mut res, seed)?;
            println!("gemm case={name} rows_diag_max_rel_diff_vs_iq_rows={diag:.3e} (diagnostic)");
            ok &= iq_nan_d(dev, &st, seed)?;
            // The host API refuses the other i-quant type on this stack.
            let stream = dev.gpu.stream();
            let mut y = DeviceBuffer::<f32>::zeroed(stream, top_k * rows)?;
            let other = if ty == GemmWeight::Iq3Xxs {
                GemmWeight::Iq4Xs
            } else {
                GemmWeight::Iq3Xxs
            };
            let refused = dev.gk.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: other,
                    w: &res.w,
                    rows_per_expert: rows,
                    act: &res.act,
                    route: &res.route,
                    input: GemmInput::Shared { top_k },
                    y: &mut y,
                },
            );
            let pass = refused.is_err();
            println!(
                "gemm case={name} other_iq_type_on_stack_refused={pass} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        if wanted("q38_iq3xxs_ragged_rows_synth") {
            let (n_exp, rows, k) = (16usize, 400usize, 2560usize);
            let st = Stack {
                name: "q38_iq3xxs_ragged_rows_synth".into(),
                ty: GemmWeight::Iq3Xxs,
                bytes: synthetic(GemmWeight::Iq3Xxs, n_exp * rows * k / 256, 0x5ea).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case(dev, &st, 0x73)?;
        }
        Ok(ok)
    }

    /// The Bolt IQ2_XXS cases: a synthetic stack of a routed gate shape — 64
    /// experts of 640 rows, K 2560, top 9 (Bolt's), every slot its token's
    /// column — and a stack of 400 rows an expert (a partial 128-row slab) at
    /// K 1280, whose odd five super-blocks a row pad each row to 332 bytes and
    /// put every other super-block at 2 mod 4. Each runs the four routings at
    /// [`TOKENS`] (the top 9 keeps 4096 tokens inside the route table's cap),
    /// the T = 1 outputs against the i-quant row core (`iq2_xxs_rows`), a NaN
    /// `d` super-block's outputs NaN, `from_ggml`'s acceptance of the type, and
    /// the host API's refusal of IQ3_XXS on the stack.
    fn iq2xxs_cases(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !["iq2xxs_gate_synth", "iq2xxs_ragged_rows_synth"]
            .iter()
            .any(|&n| wanted(n))
        {
            return Ok(true);
        }
        let accept =
            GemmWeight::from_ggml(GgmlType::IQ2_XXS).is_ok_and(|t| t == GemmWeight::Iq2Xxs);
        println!(
            "gemm case=iq2xxs_types from_ggml(iq2_xxs)={} {}",
            if accept { "accepted" } else { "REFUSED" },
            verdict(accept)
        );
        let mut ok = accept;
        {
            let (n_exp, rows, k, top_k) = (64usize, 640usize, 2560usize, 9usize);
            let st = Stack {
                name: "iq2xxs_gate_synth".into(),
                ty: GemmWeight::Iq2Xxs,
                bytes: synthetic(GemmWeight::Iq2Xxs, n_exp * rows * k / 256, 0x5f1).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(top_k),
            };
            let mut res = dev.resident_at(&st, TOKENS[TOKENS.len() - 1] * top_k)?;
            ok &= run_case(dev, &st, 0x5f2)?;
            let diag = iq_rows_diag(dev, &st, &mut res, 0x5f1)?;
            println!(
                "gemm case=iq2xxs_gate_synth rows_diag_max_rel_diff_vs_iq_rows={diag:.3e} (diagnostic)"
            );
            ok &= iq_nan_d(dev, &st, 0x5f1)?;
            // The host API refuses another i-quant type on this stack.
            let stream = dev.gpu.stream();
            let mut y = DeviceBuffer::<f32>::zeroed(stream, top_k * rows)?;
            let refused = dev.gk.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: GemmWeight::Iq3Xxs,
                    w: &res.w,
                    rows_per_expert: rows,
                    act: &res.act,
                    route: &res.route,
                    input: GemmInput::Shared { top_k },
                    y: &mut y,
                },
            );
            let pass = refused.is_err();
            println!(
                "gemm case=iq2xxs_gate_synth other_iq_type_on_stack_refused={pass} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        if wanted("iq2xxs_ragged_rows_synth") {
            let (n_exp, rows, k) = (16usize, 400usize, 1280usize);
            let st = Stack {
                name: "iq2xxs_ragged_rows_synth".into(),
                ty: GemmWeight::Iq2Xxs,
                bytes: synthetic(GemmWeight::Iq2Xxs, n_exp * rows * k / 256, 0x5f4).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case(dev, &st, 0x74)?;
        }
        Ok(ok)
    }

    /// The IQ3_S cases (the card route Qwen3.6's UD-IQ4_XS file takes on the
    /// placed load's wide arm): a synthetic gate stack of that routed shape —
    /// 256 experts of 512 rows, K 2048, top 8, every slot its token's column —
    /// a down stack of the shape the files that hold an IQ3_S down take (256
    /// experts of 2048 rows, K 512, each slot its own column), and a stack of
    /// 400 rows an expert (a partial 128-row slab) at K 1280, whose odd five
    /// super-blocks a row pad each row to 550 bytes and put every other
    /// super-block at 2 mod 4. Each runs the four routings at [`TOKENS`],
    /// `from_ggml`'s acceptance of IQ3_S beside its refusal of Q2_K, a NaN
    /// `d` super-block's outputs NaN, and the host API's refusal of the other
    /// i-quant types on the stack.
    fn iq3s_cases(dev: &Dev<'_>) -> Result<bool, GateError> {
        if ![
            "iq3s_gate_synth",
            "iq3s_down_synth",
            "iq3s_ragged_rows_synth",
        ]
        .iter()
        .any(|&n| wanted(n))
        {
            return Ok(true);
        }
        let accept = GemmWeight::from_ggml(GgmlType::IQ3_S).is_ok_and(|t| t == GemmWeight::Iq3S)
            && GemmWeight::from_ggml(GgmlType::Q2_K).is_err();
        println!(
            "gemm case=iq3s_types from_ggml(iq3_s, q2_k)={} {}",
            if accept { "accepted, refused" } else { "WRONG" },
            verdict(accept)
        );
        let mut ok = accept;
        for (name, n_exp, rows, k, input, seed) in [
            (
                "iq3s_gate_synth",
                256usize,
                512usize,
                2048usize,
                Input::Shared(8),
                0x5fbu64,
            ),
            ("iq3s_down_synth", 256, 2048, 512, Input::PerSlot(8), 0x5fd),
        ] {
            if !wanted(name) {
                continue;
            }
            let st = Stack {
                name: name.into(),
                ty: GemmWeight::Iq3S,
                bytes: synthetic(GemmWeight::Iq3S, n_exp * rows * k / 256, seed).into(),
                n_exp,
                rows,
                k,
                input,
            };
            ok &= run_case(dev, &st, seed ^ 0x61)?;
            ok &= iq_nan_d(dev, &st, seed)?;
            // The host API refuses the other i-quant types on this stack, by
            // the row size: the route is filled first, so no other check
            // refuses before the type's.
            let stream = dev.gpu.stream();
            let top_k = st.input.top_k();
            let mut res = dev.resident_at(&st, top_k)?;
            let ids = route_ids(Routing::Uniform, 1, top_k, n_exp, seed ^ 0xd1a6);
            let x = activations(k, st.input.cols(ids.len()), (seed as u32) ^ 0xd1a7);
            dev.run(&st, &mut res, &DeviceBuffer::from_host(stream, &x)?, &ids)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, top_k * rows)?;
            let mut refused = true;
            for other in [GemmWeight::Iq3Xxs, GemmWeight::Iq4Xs] {
                let outcome = dev.gk.enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty: other,
                        w: &res.w,
                        rows_per_expert: rows,
                        act: &res.act,
                        route: &res.route,
                        input: st.input.gemm(),
                        y: &mut y,
                    },
                );
                refused &= outcome.is_err_and(|e| e.to_string().contains("rows at K"));
            }
            println!(
                "gemm case={name} other_iq_types_on_stack_refused={refused} {}",
                verdict(refused)
            );
            ok &= refused;
        }
        if wanted("iq3s_ragged_rows_synth") {
            let (n_exp, rows, k) = (16usize, 400usize, 1280usize);
            let st = Stack {
                name: "iq3s_ragged_rows_synth".into(),
                ty: GemmWeight::Iq3S,
                bytes: synthetic(GemmWeight::Iq3S, n_exp * rows * k / 256, 0x5ff).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case(dev, &st, 0x75)?;
        }
        Ok(ok)
    }

    /// The IQ2_XS, IQ2_S, IQ1_S and IQ1_M cases (the routed stacks of the
    /// V4.1 sets and of Unsloth's UD-IQ1_S / UD-IQ2_M files): per type a
    /// synthetic gate stack of a routed shape — 64 experts of 640 rows, K
    /// 2560, top 9, every slot its token's column — a down stack of the
    /// shape the files that hold these types in a down take (64 experts of
    /// 2048 rows, K 512, each slot its own column) and a stack of 400 rows an
    /// expert (a partial 128-row slab) at K 1280, whose odd five super-blocks
    /// a row put every other super-block at 2 mod 4 bytes (all but IQ1_M's
    /// 56-byte block). Each runs the four routings at [`TOKENS`] through
    /// [`run_case`], whose decode is checked against ggml's own
    /// `dequantize_row_*` (the gate's port for IQ2_S, IQ1_S and IQ1_M, which
    /// `dequant_row` does not decode). The gate and down stacks also check a
    /// NaN `d` super-block's outputs NaN and the host API's refusal of the
    /// IQ3_XXS row size on the stack; `from_ggml` accepts the type.
    fn iq_card_cases(dev: &Dev<'_>) -> Result<bool, GateError> {
        let mut ok = true;
        for (tag, ty, ggml, seed) in [
            ("iq2xs", GemmWeight::Iq2Xs, GgmlType::IQ2_XS, 0x601u64),
            ("iq2s", GemmWeight::Iq2S, GgmlType::IQ2_S, 0x611),
            ("iq1s", GemmWeight::Iq1S, GgmlType::IQ1_S, 0x621),
            ("iq1m", GemmWeight::Iq1M, GgmlType::IQ1_M, 0x631),
        ] {
            let shapes = [
                ("gate", 64usize, 640usize, 2560usize, Input::Shared(9)),
                ("down", 64, 2048, 512, Input::PerSlot(8)),
                ("ragged_rows", 16, 400, 1280, Input::Shared(4)),
            ];
            if !shapes
                .iter()
                .any(|(shape, ..)| wanted(&format!("{tag}_{shape}_synth")))
            {
                continue;
            }
            let accept = GemmWeight::from_ggml(ggml).is_ok_and(|t| t == ty);
            println!(
                "gemm case={tag}_types from_ggml({tag})={} {}",
                if accept { "accepted" } else { "REFUSED" },
                verdict(accept)
            );
            ok &= accept;
            for (i, (shape, n_exp, rows, k, input)) in shapes.into_iter().enumerate() {
                let name = format!("{tag}_{shape}_synth");
                if !wanted(&name) {
                    continue;
                }
                let seed = seed + i as u64;
                let st = Stack {
                    name: name.clone(),
                    ty,
                    bytes: synthetic(ty, n_exp * rows * k / 256, seed).into(),
                    n_exp,
                    rows,
                    k,
                    input,
                };
                ok &= run_case(dev, &st, seed ^ 0x61)?;
                if shape == "ragged_rows" {
                    continue;
                }
                ok &= iq_nan_d(dev, &st, seed)?;
                // The host API refuses another i-quant type on this stack, by
                // the row size: the route is filled first, so no other check
                // refuses before the type's.
                let stream = dev.gpu.stream();
                let top_k = st.input.top_k();
                let mut res = dev.resident_at(&st, top_k)?;
                let ids = route_ids(Routing::Uniform, 1, top_k, n_exp, seed ^ 0xd1a6);
                let x = activations(k, st.input.cols(ids.len()), (seed as u32) ^ 0xd1a7);
                dev.run(&st, &mut res, &DeviceBuffer::from_host(stream, &x)?, &ids)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, top_k * rows)?;
                let outcome = dev.gk.enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty: GemmWeight::Iq3Xxs,
                        w: &res.w,
                        rows_per_expert: rows,
                        act: &res.act,
                        route: &res.route,
                        input: st.input.gemm(),
                        y: &mut y,
                    },
                );
                let refused = outcome.is_err_and(|e| e.to_string().contains("rows at K"));
                println!(
                    "gemm case={name} other_iq_type_on_stack_refused={refused} {}",
                    verdict(refused)
                );
                ok &= refused;
            }
        }
        Ok(ok)
    }

    /// `--bench-kernels`: the grouped int8 GEMM's launch cost, the eager
    /// burst and the graph replay of each arm (`bench op=` rows), timed by
    /// `tools/ref/time-gate.sh` under the machine lease. Weight and activation
    /// bytes come from [`fill_pattern`](crate::bench_arm::fill_pattern): a
    /// K-quant core's f16 scale decode takes its cheapest arm on a zero, so a
    /// zeroed buffer would read faster than the launch it prices. Nothing
    /// here asserts on the results.
    ///
    /// `--bench-arm <op>` runs only the arm whose row is `bench op=<op>`
    /// (`gemm_q4k_moe_t4096`, say) and none of the others, so every `gemm_q4k`
    /// launch of the process is that arm's — the shape a counter run (`just
    /// ncu-gpu-gemm`) filters by kernel name. A name no arm carries is a named
    /// error.
    mod bench {
        use crate::bench_arm::{
            GREPS, N, ROUNDS, burst, fill_pattern, fill_pattern_f32, print_arm, replay,
        };
        use bloomery_gpu::DeviceTensor;
        use bloomery_gpu_gates::GateError;
        use cuda_core::{CudaStream, DeviceBuffer};

        pub fn run(gpu: &bloomery_gpu::Gpu) -> Result<(), GateError> {
            println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS}");
            // `--bench-arm <op>`: that one grouped-GEMM arm and nothing else.
            let only =
                {
                    let mut args = std::env::args().skip_while(|a| a != "--bench-arm");
                    match args.next() {
                        None => None,
                        Some(_) => Some(args.next().ok_or(
                            "gate_gemm: --bench-arm wants an arm name (a `bench op=` value)",
                        )?),
                    }
                };
            if only.as_deref() == Some("zgate") {
                return if super::g32::norm_gate_case(gpu, true)? {
                    Ok(())
                } else {
                    Err(super::checks_failed())
                };
            }
            if let Some(arm) = only.as_deref() {
                return if gemm_arms(gpu, Some(arm))?
                    || iq_gemm_arms(gpu, Some(arm))?
                    || q8p_gemm_arms(gpu, Some(arm))?
                    || nvfp4_gemm_arm(gpu, arm)?
                    || nvfp4_gate_up_arm(gpu, arm)?
                {
                    Ok(())
                } else {
                    Err(format!("gate_gemm: --bench-arm {arm} names no grouped-GEMM arm").into())
                };
            }
            gemm_arms(gpu, None)?;
            iq_gemm_arms(gpu, None)?;
            q8p_gemm_arms(gpu, None)?;
            Ok(())
        }

        // The grouped int8 GEMM (`bloomery_gpu::gemm`): Qwen3-30B-A3B's routed
        // gate shape — 128 experts of 768 x 2048 Q4_K, top-8, every slot reading
        // its token's column — and the dense 2048 x 2048 Q4_K case, each at T
        // tokens up to the largest ubatch (4096: 32,768 routed slots). The route
        // table is built once per T and its launch priced on its own; the GEMM
        // arm is the GEMM alone. Beside the usual row: the
        // arithmetic rate `2 * slots * rows * K` over the graph minimum, and that
        // rate against the card's int8 dense tensor peak. `only` keeps the one
        // arm of that name; the return says whether any arm ran.
        fn gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmWeight,
            };
            let stream = gpu.stream();
            let mut ran = false;
            let gk = GemmKernels::load(gpu.context())?;
            let name = gpu.device_name()?;
            let peak = int8_peak_tops(&name);
            println!(
                "bench gemm device={name:?} int8_dense_peak_tops={}",
                peak.map_or_else(|| "?".to_string(), |p| format!("{p}"))
            );
            let sink = gpu.unlabelled_sink();
            for (label, n_exp, top_k, rows, k) in [
                ("moe", 128usize, 8usize, 768usize, 2048usize),
                ("dense", 1, 1, 2048, 2048),
            ] {
                if only.is_some_and(|o| !o.starts_with(&format!("gemm_q4k_{label}_t"))) {
                    continue;
                }
                let n_rows = n_exp * rows;
                let cols = 36 * k / 256;
                let w = DeviceTensor::<u32>::upload(
                    stream,
                    &fill_pattern(n_rows * cols),
                    n_rows,
                    cols,
                )?;
                for t in [16usize, 64, 256, 512, 1024, 2048, 4096] {
                    let op = format!("gemm_q4k_{label}_t{t}");
                    if only.is_some_and(|o| o != op) {
                        continue;
                    }
                    ran = true;
                    let n_slots = t * top_k;
                    let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                    let mut act = GemmAct::new(stream, t, k)?;
                    gpu.enqueue_quantize_gemm(&xs, t, &mut act, sink)?;
                    let ids = gemm_ids(t, top_k, n_exp);
                    let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                    let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                    let input = if n_exp == 1 {
                        gk.enqueue_route_dense(stream, n_slots, &mut route, sink)?;
                        GemmInput::PerSlot
                    } else {
                        let mut enq =
                            |s: &CudaStream| gk.enqueue_route(s, &ids_d, n_slots, &mut route, sink);
                        let e = burst(stream, &mut enq)?;
                        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                        let r = replay(stream, &g)?;
                        print_arm(
                            &format!("gemm_route_t{t}"),
                            n_slots,
                            4 * n_slots as u64,
                            g.node_count(),
                            e,
                            r,
                        );
                        GemmInput::Shared { top_k }
                    };
                    let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                    let mut enq = |s: &CudaStream| {
                        gk.enqueue_gemm(
                            s,
                            GemmArgs {
                                ty: GemmWeight::Q4K,
                                w: &w,
                                rows_per_expert: rows,
                                act: &act,
                                route: &route,
                                input,
                                y: &mut y,
                            },
                        )
                    };
                    let e = burst(stream, &mut enq)?;
                    let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                    let r = replay(stream, &g)?;
                    let mut hit = vec![false; n_exp];
                    for &i in &ids {
                        hit[i as usize] = true;
                    }
                    let experts = hit.iter().filter(|&&h| h).count();
                    let n_sb = k / 256;
                    // The weights of the experts a slot picked, the activation
                    // buffers the launch reads, and the outputs it writes.
                    let bytes = (experts * rows * 144 * n_sb
                        + t * (4 * 128 * n_sb.div_ceil(2) + 4 * 8 * n_sb + 4 * 2 * n_sb)
                        + 4 * n_slots * rows) as u64;
                    print_arm(&op, rows, bytes, g.node_count(), e, r);
                    let ops = 2.0 * (n_slots * rows * k) as f64;
                    let tops = ops / r.0 / 1e6;
                    println!(
                        "bench op={op} slots={n_slots} experts={experts} ops={ops:.0} tops_graph_min={tops:.2} \
                         pct_int8_peak={}",
                        peak.map_or_else(
                            || "?".to_string(),
                            |p| format!("{:.1}", 100.0 * tops / p)
                        )
                    );
                }
            }
            Ok(ran)
        }

        /// The i-quant arms of the two GEMM families at the Qwen3.8 card
        /// route's shapes (`gate_gemm`'s q38 cases): the routed gate — 64
        /// experts of 640 × 2560, top-10's shape — for IQ3_XXS, IQ3_S and IQ4_XS, and
        /// the routed down — 64 experts of 2560 rows × K 640, each slot its
        /// own column — for IQ4_NL. Every arm runs 4096 tokens of ten slots:
        /// the route table's cap, the heaviest launch of the shape.
        /// `only` keeps the one arm of that name; the return says whether any
        /// arm ran.
        fn iq_gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput,
                GemmKernels, GemmRoute, GemmWeight,
            };
            let stream = gpu.stream();
            let mut ran = false;
            let gk = GemmKernels::load(gpu.context())?;
            let sink = gpu.unlabelled_sink();
            for (ty, op, words, bb) in [
                (
                    GemmWeight::Iq3Xxs,
                    "gemm_iq3xxs_moe_t4096",
                    245usize,
                    98usize,
                ),
                (GemmWeight::Iq3S, "gemm_iq3s_moe_t4096", 275, 110),
                (GemmWeight::Iq4Xs, "gemm_iq4xs_moe_t4096", 340, 136),
            ] {
                if only.is_some_and(|o| o != op) {
                    continue;
                }
                ran = true;
                let (n_exp, top_k, rows, k, t) = (64usize, 10usize, 640usize, 2560usize, 4096usize);
                let n_rows = n_exp * rows;
                let w = DeviceTensor::<u32>::upload(
                    stream,
                    &fill_pattern(n_rows * words),
                    n_rows,
                    words,
                )?;
                let n_slots = t * top_k;
                let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                let mut act = GemmAct::new(stream, t, k)?;
                gpu.enqueue_quantize_gemm(&xs, t, &mut act, sink)?;
                let ids = gemm_ids(t, top_k, n_exp);
                let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                let mut rt =
                    |s: &CudaStream| gk.enqueue_route(s, &ids_d, n_slots, &mut route, sink);
                rt(stream)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut enq = |s: &CudaStream| {
                    gk.enqueue_gemm(
                        s,
                        GemmArgs {
                            ty,
                            w: &w,
                            rows_per_expert: rows,
                            act: &act,
                            route: &route,
                            input: GemmInput::Shared { top_k },
                            y: &mut y,
                        },
                    )
                };
                let e = burst(stream, &mut enq)?;
                let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                let r = replay(stream, &g)?;
                let mut hit = vec![false; n_exp];
                for &i in &ids {
                    hit[i as usize] = true;
                }
                let experts = hit.iter().filter(|&&h| h).count();
                let n_sb = k / 256;
                // The weights of the experts a slot picked, the activation
                // buffers the launch reads, and the outputs it writes.
                let bytes = (experts * rows * bb * n_sb
                    + t * (4 * 128 * n_sb.div_ceil(2) + 4 * 8 * n_sb + 4 * 2 * n_sb)
                    + 4 * n_slots * rows) as u64;
                print_arm(op, rows, bytes, g.node_count(), e, r);
            }
            {
                let op = "gemm_iq4nl_down_t4096";
                if only.is_some_and(|o| o != op) {
                    return Ok(ran);
                }
                ran = true;
                let g32 = Gemm32Kernels::load(gpu.context())?;
                let (n_exp, top_k, rows, k, t) = (64usize, 10usize, 2560usize, 640usize, 4096usize);
                let n_rows = n_exp * rows;
                // 18 bytes a block, 20 blocks: 360 bytes = 90 words a row.
                let w =
                    DeviceTensor::<u32>::upload(stream, &fill_pattern(n_rows * 90), n_rows, 90)?;
                let n_slots = t * top_k;
                let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(n_slots * k))?;
                let mut act = GemmAct32::new(stream, n_slots, k)?;
                g32.enqueue_quantize_gemm32(stream, &xs, n_slots, &mut act, sink)?;
                let ids = gemm_ids(t, top_k, n_exp);
                let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                gk.enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut enq = |s: &CudaStream| {
                    g32.enqueue_gemm32(
                        s,
                        Gemm32Args {
                            w: Gemm32Weight::Iq4NlFile(&w),
                            rows_per_expert: rows,
                            act: &act,
                            route: &route,
                            input: GemmInput::PerSlot,
                            y: &mut y,
                        },
                    )
                };
                let e = burst(stream, &mut enq)?;
                let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                let r = replay(stream, &g)?;
                let mut hit = vec![false; n_exp];
                for &i in &ids {
                    hit[i as usize] = true;
                }
                let experts = hit.iter().filter(|&&h| h).count();
                let bytes = (experts * rows * 18 * (k / 32)
                    + n_slots * (4 * 16 * act.steps() + 4 * 2 * act.steps())
                    + 4 * n_slots * rows) as u64;
                print_arm(op, rows, bytes, g.node_count(), e, r);
            }
            Ok(ran)
        }

        /// The dense Q8_0-plane GEMM (`gemm_q8_0p`) at Qwen3.8's dense prompt
        /// projections, `rows × K` over the one-expert table: a linear-attention
        /// layer's `qkv` (10240 × 2560), `z` (6144 × 2560) and `out` (2560 ×
        /// 6144, the attention output's shape too); an attention layer's `q`
        /// (12288 × 2560) and `kv` (512 × 2560); the shared expert's gate·up
        /// `shgu` (640 × 2560) and down `shdown` (2560 × 640); a hyper-connection
        /// mix's down `hcdown` (320 × 10240) and up `hcup` (10240 × 320). Each at
        /// T ∈ {16, 512, 2048} columns: a unit just past the gemv's eight
        /// columns and the prompt's two ubatch sizes. Then K 2560 stacks around
        /// one resident-block wave read from the loaded slab entry, a block per
        /// 128-row slab and tile: at T 16 (one tile) one block under (`wavem1`),
        /// one wave (`wave`) and one block over (`wavep1`); at T 65 (two tiles,
        /// an even grid) the last grid below the wave (`wave2m`), the first at
        /// or above it (`wave2`) and two blocks over (`wave2p`). Beside the
        /// usual row, the rate `2 * T * rows * K` over
        /// the graph minimum. `only` keeps the one arm of that name; the return
        /// says whether any arm ran.
        fn q8p_gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmKernels,
                GemmRoute,
            };
            let stream = gpu.stream();
            let mut ran = false;
            let sink = gpu.unlabelled_sink();
            let (gk, g32) = (
                GemmKernels::load(gpu.context())?,
                Gemm32Kernels::load(gpu.context())?,
            );
            let wave = usize::try_from(g32.q8_0p_wave())?;
            let wave2 = wave.div_ceil(2);
            for (label, rows, k, ts) in [
                ("qkv", 10240usize, 2560usize, &[16usize, 512, 2048][..]),
                ("z", 6144, 2560, &[16, 512, 2048]),
                ("out", 2560, 6144, &[16, 512, 2048]),
                ("q", 12288, 2560, &[16, 512, 2048]),
                ("kv", 512, 2560, &[16, 512, 2048]),
                ("shgu", 640, 2560, &[16, 512, 2048]),
                ("shdown", 2560, 640, &[16, 512, 2048]),
                ("hcdown", 320, 10240, &[16, 512, 2048]),
                ("hcup", 10240, 320, &[16, 512, 2048]),
                ("wavem1", 128 * (wave - 1), 2560, &[16]),
                ("wave", 128 * wave, 2560, &[16]),
                ("wavep1", 128 * (wave + 1), 2560, &[16]),
                ("wave2m", 128 * (wave2 - 1), 2560, &[65]),
                ("wave2", 128 * wave2, 2560, &[65]),
                ("wave2p", 128 * (wave2 + 1), 2560, &[65]),
            ] {
                let prefix = format!("gemm_q8_0p_{label}_t");
                if only.is_some_and(|o| !o.starts_with(&prefix)) {
                    continue;
                }
                let kb = k / 32;
                let qs =
                    DeviceTensor::<u32>::upload(stream, &fill_pattern(rows * k / 4), rows, k / 4)?;
                // Finite f16 scales in [2^-7, 2^-6): the pattern's low bits as the
                // mantissa.
                let d_bits: Vec<u16> = fill_pattern(rows * kb)
                    .into_iter()
                    .map(|w| 0x2000 | (w & 0x03ff) as u16)
                    .collect();
                let d = DeviceTensor::<u16>::upload(stream, &d_bits, rows, kb)?;
                for &t in ts {
                    let op = format!("{prefix}{t}");
                    if only.is_some_and(|o| o != op) {
                        continue;
                    }
                    ran = true;
                    let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                    let mut act = GemmAct32::new(stream, t, k)?;
                    g32.enqueue_quantize_gemm32(stream, &xs, t, &mut act, sink)?;
                    let mut route = GemmRoute::new(stream, t, 1)?;
                    gk.enqueue_route_dense(stream, t, &mut route, sink)?;
                    let mut y = DeviceBuffer::<f32>::zeroed(stream, t * rows)?;
                    let mut enq = |s: &CudaStream| {
                        g32.enqueue_gemm32(
                            s,
                            Gemm32Args {
                                w: Gemm32Weight::Q8_0Plane { qs: &qs, d: &d },
                                rows_per_expert: rows,
                                act: &act,
                                route: &route,
                                input: GemmInput::PerSlot,
                                y: &mut y,
                            },
                        )
                    };
                    let e = burst(stream, &mut enq)?;
                    let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                    let r = replay(stream, &g)?;
                    // The planes, the activation planes the launch reads, and the
                    // outputs it writes.
                    let bytes = (rows * (k + 2 * kb)
                        + t * (4 * 16 * act.steps() + 4 * 2 * act.steps())
                        + 4 * t * rows) as u64;
                    print_arm(&op, rows, bytes, g.node_count(), e, r);
                    let ops = 2.0 * (t * rows * k) as f64;
                    println!(
                        "bench op={op} slots={t} k={k} ops={ops:.0} tops_graph_min={:.2}",
                        ops / r.0 / 1e6
                    );
                }
            }
            Ok(ran)
        }

        fn nvfp4_gemm_arm(gpu: &bloomery_gpu::Gpu, arm: &str) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmKernels,
                GemmRoute, SwigluSlots,
            };
            use bloomery_gpu::mxfp4_sel::Nvfp4Stack;
            // `swiglu`: the up GEMM with the card slots' SwiGLU into the down's
            // activations, `Some(true)` folded into one launch
            // (`enqueue_up_swiglu32`), `Some(false)` the up GEMM and then
            // `swiglu_quant32_sel`.
            let (rows, k, per_slot, tokens, swiglu) = match arm {
                "gemm_nvfp4_gate_t2048" => (640usize, 2560usize, false, 2048usize, None),
                "gemm_nvfp4_down_t2048" => (2560, 640, true, 2048, None),
                "gemm_nvfp4_gate_t4096" => (640, 2560, false, 4096, None),
                "gemm_nvfp4_down_t4096" => (2560, 640, true, 4096, None),
                "gemm_nvfp4_upswiglu_t4096" => (640, 2560, false, 4096, Some(true)),
                "gemm_nvfp4_upsep_t4096" => (640, 2560, false, 4096, Some(false)),
                _ => return Ok(false),
            };
            let (experts, top_k) = (512usize, 10usize);
            let slots = tokens * top_k;
            let cols = if per_slot { slots } else { tokens };
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let g32 = Gemm32Kernels::load(gpu.context())?;
            let gk = GemmKernels::load(gpu.context())?;
            let qs = DeviceTensor::upload(
                stream,
                &fill_pattern(experts * rows * k / 8),
                experts * rows,
                k / 8,
            )?;
            let se: Vec<u32> = fill_pattern(experts * rows * k / 64)
                .into_iter()
                .map(|v| v & 0x7e7e7e7e)
                .collect();
            let e = DeviceTensor::upload(stream, &se, experts * rows, k / 64)?;
            let scales = DeviceTensor::upload(stream, &vec![0.015625f32; experts], experts, 1)?;
            let xs = DeviceBuffer::from_host(stream, &fill_pattern_f32(cols * k))?;
            let mut act = GemmAct32::new_routed(stream, cols, k)?;
            g32.enqueue_quantize_gemm32(stream, &xs, cols, &mut act, sink)?;
            let ids = DeviceBuffer::from_host(stream, &gemm_ids(tokens, top_k, experts))?;
            let mut route = GemmRoute::new(stream, slots, experts)?;
            gk.enqueue_route(stream, &ids, slots, &mut route, sink)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, slots * rows)?;
            let g = DeviceBuffer::from_host(stream, &fill_pattern_f32(slots * rows))?;
            let sel = DeviceBuffer::<u32>::zeroed(stream, slots)?;
            let mut out_act = GemmAct32::new_routed(stream, slots, rows)?;
            let mut enq = |s: &CudaStream| {
                let up = Gemm32Args {
                    w: Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                        qs: &qs,
                        e: &e,
                        scales: &scales,
                    }),
                    rows_per_expert: rows,
                    act: &act,
                    route: &route,
                    input: if per_slot {
                        GemmInput::PerSlot
                    } else {
                        GemmInput::Shared { top_k }
                    },
                    y: &mut y,
                };
                match swiglu {
                    None => g32.enqueue_gemm32(s, up),
                    Some(true) => g32.enqueue_up_swiglu32(
                        s,
                        up,
                        &g,
                        SwigluSlots::Card {
                            sel: &sel,
                            n_card: experts,
                        },
                        &mut out_act,
                        sink,
                    ),
                    Some(false) => {
                        g32.enqueue_gemm32(s, up)?;
                        g32.enqueue_swiglu_quant32_sel(
                            s,
                            &g,
                            &y,
                            &sel,
                            experts,
                            slots,
                            &mut out_act,
                            sink,
                        )
                    }
                }
            };
            let eager = burst(stream, &mut enq)?;
            let graph = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            let timed = replay(stream, &graph)?;
            let bytes = (experts * rows * k * 9 / 16 + cols * k + slots * rows * 4) as u64;
            print_arm(arm, rows, bytes, graph.node_count(), eager, timed);
            if swiglu.is_some() {
                use bloomery_gpu_gates::Fnv1a64;
                println!(
                    "bench op={arm} slots={slots} k={k} experts={experts} act_q_fnv={:016x} \
                     act_d_fnv={:016x}",
                    Fnv1a64::default()
                        .u32s(&out_act.q().to_host_vec(stream)?)
                        .value(),
                    Fnv1a64::default()
                        .f32s(&out_act.d().to_host_vec(stream)?)
                        .value()
                );
                if gpu.take_fault()?.is_some() {
                    return Err("NVFP4 SwiGLU benchmark raised a fault".into());
                }
                return Ok(true);
            }
            let out = y.to_host_vec(stream)?;
            let invalid = out.iter().filter(|v| !v.is_finite()).count();
            let norm: f64 = out
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            println!(
                "bench op={arm} slots={slots} k={k} experts={experts} invalid={invalid} l2={norm:.9} \
                 y_fnv={:016x} sample={:?}",
                bloomery_gpu_gates::Fnv1a64::default().f32s(&out).value(),
                &out[..8]
            );
            if invalid != 0 || gpu.take_fault()?.is_some() {
                return Err("NVFP4 benchmark produced invalid output".into());
            }
            Ok(true)
        }

        /// A routed block's gate, up and SwiGLU at Qwen3.8's prompt shape (512
        /// NVFP4 experts of 640 x 2560, top-10, 4,096 tokens): `_split` the
        /// gate and up GEMMs and `swiglu_quant32_sel`, the other
        /// `enqueue_gate_up_sel32` into its own scratch (one launch that packs
        /// the FP4 on a native build). Each arm prints its activations' hash
        /// (the `nvfp4_gate_up` case holds the two forms to each other).
        fn nvfp4_gate_up_arm(gpu: &bloomery_gpu::Gpu, arm: &str) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                GateUp32Args, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput,
                GemmKernels, GemmRoute,
            };
            use bloomery_gpu::mxfp4_sel::Nvfp4Stack;
            let split = match arm {
                "gemm_nvfp4_gate_up_t4096" => false,
                "gemm_nvfp4_gate_up_split_t4096" => true,
                _ => return Ok(false),
            };
            let (rows, k, tokens, experts, top_k) = (640usize, 2560usize, 4096usize, 512, 10);
            let slots = tokens * top_k;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let g32 = Gemm32Kernels::load(gpu.context())?;
            let gk = GemmKernels::load(gpu.context())?;
            let se: Vec<u32> = fill_pattern(experts * rows * k / 64)
                .into_iter()
                .map(|v| v & 0x7e7e7e7e)
                .collect();
            let e = DeviceTensor::upload(stream, &se, experts * rows, k / 64)?;
            let scales = DeviceTensor::upload(stream, &vec![0.015625f32; experts], experts, 1)?;
            let gq = DeviceTensor::upload(
                stream,
                &fill_pattern(experts * rows * k / 8),
                experts * rows,
                k / 8,
            )?;
            let uq = DeviceTensor::upload(
                stream,
                &fill_pattern(experts * rows * k / 8 + 1)[1..],
                experts * rows,
                k / 8,
            )?;
            let gate = Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                qs: &gq,
                e: &e,
                scales: &scales,
            });
            let up = Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                qs: &uq,
                e: &e,
                scales: &scales,
            });
            let xs = DeviceBuffer::from_host(stream, &fill_pattern_f32(tokens * k))?;
            let mut x = GemmAct32::new_routed(stream, tokens, k)?;
            g32.enqueue_quantize_gemm32(stream, &xs, tokens, &mut x, sink)?;
            let ids = gemm_ids(tokens, top_k, experts);
            let ids_d = DeviceBuffer::from_host(stream, &ids)?;
            let mut route = GemmRoute::new(stream, slots, experts)?;
            gk.enqueue_route(stream, &ids_d, slots, &mut route, sink)?;
            let mut g = DeviceBuffer::<f32>::zeroed(stream, slots * rows)?;
            let mut u = DeviceBuffer::<f32>::zeroed(stream, slots * rows)?;
            let mut act = if split {
                GemmAct32::new_routed(stream, slots, rows)?
            } else {
                GemmAct32::new_gate_up_output(stream, slots, rows)?
            };
            let mut enq = |s: &CudaStream, act: &mut GemmAct32| {
                if !split {
                    return g32.enqueue_gate_up_sel32(
                        s,
                        GateUp32Args {
                            w: [gate, up],
                            rows_per_expert: rows,
                            x: &x,
                            route: &route,
                            input: GemmInput::Shared { top_k },
                            sel: &ids_d,
                            n_card: experts,
                            g: &mut g,
                            u: &mut u,
                            act,
                        },
                        sink,
                    );
                }
                for (w, y) in [(gate, &mut g), (up, &mut u)] {
                    g32.enqueue_gemm32(
                        s,
                        Gemm32Args {
                            w,
                            rows_per_expert: rows,
                            act: &x,
                            route: &route,
                            input: GemmInput::Shared { top_k },
                            y,
                        },
                    )?;
                }
                g32.enqueue_swiglu_quant32_sel(s, &g, &u, &ids_d, experts, slots, act, sink)
            };
            let eager = burst(stream, &mut |s| enq(s, &mut act))?;
            let graph = gpu.capture(|s| (0..N).try_for_each(|_| enq(s, &mut act)))?;
            let timed = replay(stream, &graph)?;
            let bytes =
                (2 * experts * rows * k * 9 / 16 + tokens * k + slots * rows * 5 / 4) as u64;
            print_arm(arm, rows, bytes, graph.node_count(), eager, timed);
            // The codes the down reads: the FP4 ones on a native build, the Q8 ones otherwise.
            let codes = act.fp4_planes().map_or(act.q(), |(codes, _, _)| codes);
            println!(
                "bench op={arm} slots={slots} k={k} experts={experts} codes_fnv={:016x}",
                bloomery_gpu_gates::Fnv1a64::default()
                    .u32s(&codes.to_host_vec(stream)?)
                    .value()
            );
            if gpu.take_fault()?.is_some() {
                return Err("NVFP4 gate and up benchmark raised a fault".into());
            }
            Ok(true)
        }

        /// The int8 dense tensor peak of the card named `name`, in TOPS — GA102
        /// whitepaper figures, the denominator the prefill literature report uses —
        /// or `None` for a card not listed.
        fn int8_peak_tops(name: &str) -> Option<f64> {
            if name.contains("A6000") {
                Some(309.7)
            } else if name.contains("3090") {
                Some(284.0)
            } else {
                None
            }
        }

        /// Expert ids for `t` tokens of `top_k` distinct experts out of `n_exp`,
        /// drawn from a fixed LCG: the bench's uniform routing.
        fn gemm_ids(t: usize, top_k: usize, n_exp: usize) -> Vec<u32> {
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let mut ids = Vec::with_capacity(t * top_k);
            for _ in 0..t {
                let mut pick: Vec<u32> = Vec::with_capacity(top_k);
                while pick.len() < top_k {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let e = ((s >> 33) % n_exp as u64) as u32;
                    if !pick.contains(&e) {
                        pick.push(e);
                    }
                }
                ids.extend(pick);
            }
            ids
        }
    }

    /// The 32-value family (`Gemm32Kernels`): module doc, last part.
    mod g32 {
        use super::{
            Lcg, ROUTINGS, Routing, SENT, TOKENS, pad, route_ids, route_ref, row_checked,
            synthetic, wanted,
        };
        use bloomery_gpu::gemm::{
            GEMM_BN, GateUp32Args, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32,
            GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmWeight, SwigluSlots,
        };
        use bloomery_gpu::hybrid::HOST;
        use bloomery_gpu::kquant::{Act, act::silu_ik};
        use bloomery_gpu::mxfp4::{Nvfp4, repack, ue4m3_half};
        use bloomery_gpu::mxfp4_sel::Nvfp4Stack;
        use bloomery_gpu::q5::{Q8Blocks32, pack_q5_0};
        use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE};
        use bloomery_gpu_gates::gemm32::{
            HostAct, Planes32, dot32, dot32_nv, dot32_nv_fp4, gate_up_fp4, host_act,
        };
        use bloomery_gpu_gates::rounding::gamma;
        use bloomery_gpu_gates::swiglu_host;
        use bloomery_gpu_gates::{
            Fnv1a64, GateError, activations, bits_equal, bytes_to_words, verdict,
        };
        use cuda_core::{CudaStream, DeviceBuffer};
        use gguf::iq_tables::KVALUES_IQ4NL;
        use gguf::quant::{GgmlType, KVALUES_MXFP4, dequant_row, half_to_f32};
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// `each` over the slots `0..slots`, spread over the machine's threads
        /// (at most 32), each thread folding into its own `init()`. Join every
        /// worker before returning an error if any worker panicked.
        fn par_slots<T: Send>(
            slots: usize,
            init: impl Fn() -> T + Sync,
            each: impl Fn(usize, &mut T) + Sync,
        ) -> Result<Vec<T>, GateError> {
            let next = AtomicUsize::new(0);
            let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads)
                    .map(|_| {
                        sc.spawn(|| {
                            let mut t = init();
                            loop {
                                let s = next.fetch_add(1, Ordering::Relaxed);
                                if s >= slots {
                                    break;
                                }
                                each(s, &mut t);
                            }
                            t
                        })
                    })
                    .collect();
                let mut parts = Vec::with_capacity(threads);
                let mut panicked = false;
                for h in hs {
                    match h.join() {
                        Ok(part) => parts.push(part),
                        Err(_) => panicked = true,
                    }
                }
                if panicked {
                    Err("reference thread panicked".into())
                } else {
                    Ok(parts)
                }
            })
        }

        /// K per dense case: 96 and 352 end in a one-block step (K/32 odd),
        /// 320 and 640 are Qwen3.8's HC up and shared down, not multiples of
        /// 256; 2560, 6144 and 10240 its hidden, GDN value and HC widths.
        const KS: [usize; 7] = [96, 320, 352, 640, 2560, 6144, 10240];
        /// Columns per dense case.
        const MS: [usize; 5] = [1, 8, 9, 512, 4096];
        /// Rows of a dense case: two full 128-row slabs and one of 16.
        const ROWS: usize = 272;

        /// The four layouts under test.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Lay {
            Q8Plane,
            Q8File,
            Iq4Nl,
            Q51,
        }

        /// A synthetic stack of `n_rows` rows of `k` values, decoded to what
        /// the kernels multiply: per block `d`, `m` (Q5_1) and per value the
        /// code; `bytes` the file's block stream.
        struct Wts {
            k: usize,
            bytes: Vec<u8>,
            d: Vec<f32>,
            m: Vec<f32>,
            q: Vec<i8>,
        }

        /// A finite f16 of exponent 3..=9 (2^-12 .. 2^-5), sign from `neg`.
        fn f16_bits(g: &mut Lcg, neg: bool) -> u16 {
            let e = 3 + g.below(7) as u16;
            let bits = (e << 10) | (g.next() as u16 & 0x3ff);
            if neg { bits | 0x8000 } else { bits }
        }

        /// `n_rows` rows of Q8_0 (`q51` false) or Q5_1 blocks from `seed`,
        /// decoded; the decode is checked against ggml's `dequant_row` (Q8_0
        /// bit for bit, Q5_1 within one rounding of its fma).
        fn weights(
            q51: bool,
            n_rows: usize,
            k: usize,
            seed: u64,
        ) -> Result<(Wts, bool), GateError> {
            let mut g = Lcg(seed);
            let kb = k / 32;
            let bpb = if q51 { 24 } else { 34 };
            let mut bytes = vec![0u8; n_rows * kb * bpb];
            let (mut d, mut m, mut q) = (
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * k),
            );
            for blk in bytes.chunks_exact_mut(bpb) {
                for b in blk.iter_mut() {
                    *b = g.next() as u8;
                }
                let db = f16_bits(&mut g, false);
                blk[0..2].copy_from_slice(&db.to_le_bytes());
                d.push(half_to_f32(db));
                if q51 {
                    let neg = g.next() & 1 == 1;
                    let mb = f16_bits(&mut g, neg);
                    blk[2..4].copy_from_slice(&mb.to_le_bytes());
                    m.push(half_to_f32(mb));
                    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                    let qs = &blk[8..24];
                    let mut v = [0i8; 32];
                    for (j, &b) in qs.iter().enumerate() {
                        v[j] = ((b & 0x0f) | ((((qh >> j) & 1) as u8) << 4)) as i8;
                        v[16 + j] = ((b >> 4) | ((((qh >> (16 + j)) & 1) as u8) << 4)) as i8;
                    }
                    q.extend_from_slice(&v);
                } else {
                    m.push(0.0);
                    q.extend(blk[2..34].iter().map(|&b| b as i8));
                }
            }
            let ty = if q51 { GgmlType::Q5_1 } else { GgmlType::Q8_0 };
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(ty, &bytes[r * kb * bpb..(r + 1) * kb * bpb], &mut want)?;
                for (v, &w) in want.iter().enumerate() {
                    let b = r * kb + v / 32;
                    let mine = f64::from(d[b]) * f64::from(q[r * k + v]) + f64::from(m[b]);
                    decode_ok &= if q51 {
                        (mine - f64::from(w)).abs() <= gamma(1) * mine.abs() + 1e-38
                    } else {
                        mine as f32 == w
                    };
                }
            }
            Ok((Wts { k, bytes, d, m, q }, decode_ok))
        }

        /// `n_rows` rows of IQ4_NL blocks from `seed`, decoded: per block a
        /// finite f16 `d` and 16 random code bytes, their low nibbles values
        /// `0..16` and their high nibbles `16..32`, each a `kvalues_iq4nl`
        /// entry; the decode is checked against ggml's `dequant_row` bit for
        /// bit (d·kv is one f32 product on both sides).
        fn weights_iq4nl(n_rows: usize, k: usize, seed: u64) -> Result<(Wts, bool), GateError> {
            let mut g = Lcg(seed);
            let kb = k / 32;
            let mut bytes = vec![0u8; n_rows * kb * 18];
            let (mut d, mut q) = (
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * k),
            );
            for blk in bytes.as_chunks_mut::<18>().0 {
                for b in blk.iter_mut() {
                    *b = g.next() as u8;
                }
                let db = f16_bits(&mut g, false);
                blk[0..2].copy_from_slice(&db.to_le_bytes());
                d.push(half_to_f32(db));
                for j in 0..16 {
                    q.push(KVALUES_IQ4NL[usize::from(blk[2 + j] & 0x0f)]);
                }
                for j in 0..16 {
                    q.push(KVALUES_IQ4NL[usize::from(blk[2 + j] >> 4)]);
                }
            }
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(
                    GgmlType::IQ4_NL,
                    &bytes[r * kb * 18..(r + 1) * kb * 18],
                    &mut want,
                )?;
                for (v, &w) in want.iter().enumerate() {
                    let b = r * kb + v / 32;
                    let mine = f64::from(d[b]) * f64::from(q[r * k + v]);
                    decode_ok &= mine as f32 == w;
                }
            }
            Ok((
                Wts {
                    k,
                    bytes,
                    d,
                    m: vec![0.0; n_rows * kb],
                    q,
                },
                decode_ok,
            ))
        }

        /// A synthetic NVFP4 stack, decoded: `bytes` the file's block stream,
        /// `q` the rows' `k` codes in value order, `ds` their `k/16`
        /// sub-block scales, `scale` the tensor's.
        struct WtsNv {
            bytes: Vec<u8>,
            scale: f32,
            q: Vec<i8>,
            ds: Vec<f32>,
        }

        /// `n_rows` rows of NVFP4 blocks from `seed`: per 64-value block four
        /// scale bytes then 32 random code bytes. Every other block's scales
        /// are the ladder 0x70 0x50 0x30 0x10 — every pair 16× apart, so a
        /// swapped sub-block scale moves its unit's outputs far outside the
        /// band; the rest are random bytes UE4M3 takes (bit 7 clear, not
        /// 0x7F). Decoded to the E2M1 int8 codes in value order (a sub-block
        /// value `j < 8` the low nibble of its code byte `j`, `j + 8` the
        /// high) and the halved UE4M3 scales; the decode is checked against
        /// ggml's `dequant_row` bit for bit (`d_s·code` exact in f32 on both
        /// sides). The tensor `scale` is 2^-6, not 1, so a store without it
        /// fails the contract.
        fn weights_nvfp4(n_rows: usize, k: usize, seed: u64) -> Result<(WtsNv, bool), GateError> {
            let mut g = Lcg(seed);
            let nblk = k / 64;
            let mut bytes = vec![0u8; n_rows * nblk * 36];
            let (mut q, mut ds) = (
                Vec::with_capacity(n_rows * k),
                Vec::with_capacity(n_rows * k / 16),
            );
            for (bi, blk) in bytes.as_chunks_mut::<36>().0.iter_mut().enumerate() {
                for b in blk[4..].iter_mut() {
                    *b = g.next() as u8;
                }
                if bi & 1 == 0 {
                    blk[..4].copy_from_slice(&[0x70, 0x50, 0x30, 0x10]);
                } else {
                    for b in blk[..4].iter_mut() {
                        *b = (g.next() as u8 & 0x7f).min(0x7e);
                    }
                }
                for s in 0..4 {
                    ds.push(ue4m3_half(u32::from(blk[s])));
                    for j in 0..8 {
                        q.push(KVALUES_MXFP4[usize::from(blk[4 + 8 * s + j] & 0x0f)]);
                    }
                    for j in 0..8 {
                        q.push(KVALUES_MXFP4[usize::from(blk[4 + 8 * s + j] >> 4)]);
                    }
                }
            }
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(
                    GgmlType::NVFP4,
                    &bytes[r * nblk * 36..(r + 1) * nblk * 36],
                    &mut want,
                )?;
                for (v, &w) in want.iter().enumerate() {
                    decode_ok &= ds[r * k / 16 + v / 16] * f32::from(q[r * k + v]) == w;
                }
            }
            Ok((
                WtsNv {
                    bytes,
                    scale: 0.015_625,
                    q,
                    ds,
                },
                decode_ok,
            ))
        }

        /// The device planes of the first `n` columns equal the host's bit
        /// for bit, padding included.
        fn planes_equal(
            act: &GemmAct32,
            n: usize,
            want: &Planes32,
            stream: &cuda_core::CudaStream,
        ) -> Result<bool, GateError> {
            let st = act.steps();
            let q = act.q().to_host_vec(stream)?;
            let d = act.d().to_host_vec(stream)?;
            let s = act.s().to_host_vec(stream)?;
            Ok(q[..n * 16 * st] == want.0[..]
                && nan_bits_equal(&d[..n * 2 * st], &want.1)
                && s[..n * 2 * st] == want.2[..])
        }

        /// Two planes equal: each pair the same bits, or both NaN (a refused
        /// block's scale or output, whatever its payload).
        fn nan_bits_equal(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
        }

        /// One output of stack row `row` against column `col` three ways
        /// ([`dot32`]): the contract's bits, the f64 value and its band's
        /// magnitude.
        fn row_dot(w: &Wts, row: usize, a: &HostAct, col: usize, mins: bool) -> (f32, f64, f64) {
            let (k, kb) = (w.k, w.k / 32);
            dot32(
                &w.q[row * k..(row + 1) * k],
                &w.d[row * kb..(row + 1) * kb],
                &w.m[row * kb..(row + 1) * kb],
                a,
                col,
                mins,
            )
        }

        /// One run's reference check: outputs checked, outside the band,
        /// off the contract's bits, host slots written; the worst error over
        /// its band and the first failure.
        struct Checked {
            checked: usize,
            band_off: usize,
            bits_off: usize,
            host_bad: usize,
            worst: f64,
            first: Option<String>,
        }

        /// The reference check of one run over `slots` slots: slot `s` on
        /// weight rows `e(s)·rows ..` (`None`: a host slot, whose outputs must
        /// still read [`SENT`]), column `col(s)`.
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn check(
            w: &Wts,
            mins: bool,
            rows: usize,
            a: &HostAct,
            expert: &(dyn Fn(usize) -> Option<usize> + Sync),
            col: &(dyn Fn(usize) -> usize + Sync),
            y: &[f32],
            slots: usize,
        ) -> Result<Checked, GateError> {
            let kb = w.k / 32;
            let parts = par_slots(
                slots,
                || Checked {
                    checked: 0,
                    band_off: 0,
                    bits_off: 0,
                    host_bad: 0,
                    worst: 0.0,
                    first: None,
                },
                |s, c: &mut Checked| {
                    let out = &y[s * rows..(s + 1) * rows];
                    let Some(e) = expert(s) else {
                        if out.iter().any(|v| v.to_bits() != SENT.to_bits()) {
                            c.host_bad += 1;
                            c.first
                                .get_or_insert_with(|| format!("host slot {s} written"));
                        }
                        return;
                    };
                    for r in (0..rows).filter(|&r| row_checked(slots, s, r)) {
                        let (want, y64, mag) = row_dot(w, e * rows + r, a, col(s), mins);
                        let got = out[r];
                        let band = gamma(kb + 3) * mag;
                        let err = (f64::from(got) - y64).abs();
                        c.checked += 1;
                        c.worst = c.worst.max(if band > 0.0 { err / band } else { err });
                        let within = err <= band;
                        if !within {
                            c.band_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: got {got:e} f64 {y64:e} band {band:.3e}")
                            });
                        }
                        if got.to_bits() != want.to_bits() {
                            c.bits_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: got {got:e} contract {want:e}")
                            });
                        }
                    }
                },
            )?;
            let mut all = Checked {
                checked: 0,
                band_off: 0,
                bits_off: 0,
                host_bad: 0,
                worst: 0.0,
                first: None,
            };
            for p in parts {
                all.checked += p.checked;
                all.band_off += p.band_off;
                all.bits_off += p.bits_off;
                all.host_bad += p.host_bad;
                all.worst = all.worst.max(p.worst);
                all.first = all.first.or(p.first);
            }
            Ok(all)
        }

        /// The dot-injected reference check (the NVFP4 planes' `dot32_nv`):
        /// slot `s` on
        /// weight rows `e(s)·rows ..` (`None`: a host slot, whose outputs must
        /// still read [`SENT`]), column `col(s)`, one output's three ways
        /// (`dot(row, col)`: the contract's bits, the f64 value and its band's
        /// magnitude), every checked output inside `γ(roundings)` of its magnitude
        /// (`dot32`: the plain families' `K/32 + 3`, `dot32_nv`: `2·K/32 + 4`).
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn check_dot(
            roundings: usize,
            rows: usize,
            expert: &(dyn Fn(usize) -> Option<usize> + Sync),
            col: &(dyn Fn(usize) -> usize + Sync),
            dot: &(dyn Fn(usize, usize) -> (f32, f64, f64) + Sync),
            y: &[f32],
            slots: usize,
        ) -> Result<Checked, GateError> {
            let parts = par_slots(
                slots,
                || Checked {
                    checked: 0,
                    band_off: 0,
                    bits_off: 0,
                    host_bad: 0,
                    worst: 0.0,
                    first: None,
                },
                |s, c: &mut Checked| {
                    let out = &y[s * rows..(s + 1) * rows];
                    let Some(e) = expert(s) else {
                        if out.iter().any(|v| v.to_bits() != SENT.to_bits()) {
                            c.host_bad += 1;
                            c.first
                                .get_or_insert_with(|| format!("host slot {s} written"));
                        }
                        return;
                    };
                    for r in (0..rows).filter(|&r| row_checked(slots, s, r)) {
                        let (want, y64, mag) = dot(e * rows + r, col(s));
                        let got = out[r];
                        let band = gamma(roundings) * mag;
                        let err = (f64::from(got) - y64).abs();
                        c.checked += 1;
                        c.worst = c.worst.max(if band > 0.0 { err / band } else { err });
                        let within = err <= band;
                        if !within {
                            c.band_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: got {got:e} f64 {y64:e} band {band:.3e}")
                            });
                        }
                        if got.to_bits() != want.to_bits() {
                            c.bits_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: got {got:e} contract {want:e}")
                            });
                        }
                    }
                },
            )?;
            let mut all = Checked {
                checked: 0,
                band_off: 0,
                bits_off: 0,
                host_bad: 0,
                worst: 0.0,
                first: None,
            };
            for p in parts {
                all.checked += p.checked;
                all.band_off += p.band_off;
                all.bits_off += p.bits_off;
                all.host_bad += p.host_bad;
                all.worst = all.worst.max(p.worst);
                all.first = all.first.or(p.first);
            }
            Ok(all)
        }
        /// The device side: both kernel families' modules and the engine.
        struct Dev<'g> {
            gpu: &'g Gpu,
            gk: &'g GemmKernels,
            g32: Gemm32Kernels,
        }

        /// A stack resident in every layout it has: a Q8_0 stack as the
        /// planes and as the file's words, a Q5_1 stack as the file's words.
        struct Res {
            plane: Option<(DeviceTensor<u32>, DeviceTensor<u16>)>,
            file: DeviceTensor<u32>,
        }

        impl Res {
            /// `no_plane` for the layouts without q8f32 planes (Q5_1,
            /// IQ4_NL): the stack goes up as the file's words alone.
            fn new(
                dev: &Dev<'_>,
                w: &Wts,
                no_plane: bool,
                n_rows: usize,
            ) -> Result<Res, GateError> {
                let stream = dev.gpu.stream();
                let kb = w.k / 32;
                let plane = if no_plane {
                    None
                } else {
                    let qs: Vec<u32> =
                        w.q.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|c| {
                                u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])
                            })
                            .collect();
                    let d: Vec<u16> = w
                        .bytes
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    Some((
                        DeviceTensor::upload(stream, &qs, n_rows, w.k / 4)?,
                        DeviceTensor::upload(stream, &d, n_rows, kb)?,
                    ))
                };
                let mut words = bytes_to_words(&w.bytes);
                words.resize(words.len().div_ceil(n_rows) * n_rows, 0);
                let cols = words.len() / n_rows;
                Ok(Res {
                    plane,
                    file: DeviceTensor::upload(stream, &words, n_rows, cols)?,
                })
            }

            fn weight(&self, lay: Lay) -> Result<Gemm32Weight<'_>, GateError> {
                Ok(match lay {
                    Lay::Q8Plane => {
                        let (qs, d) = self.plane.as_ref().ok_or("a Q5_1 stack has no planes")?;
                        Gemm32Weight::Q8_0Plane { qs, d }
                    }
                    Lay::Q8File => Gemm32Weight::Q8_0File(&self.file),
                    Lay::Iq4Nl => Gemm32Weight::Iq4NlFile(&self.file),
                    Lay::Q51 => Gemm32Weight::Q5_1File(&self.file),
                })
            }
        }

        /// One GEMM over the table `route` last filled with `n_slots` slots,
        /// `y` set to [`SENT`] first; `y`'s first `n_slots·rows` values back.
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn gemm(
            dev: &Dev<'_>,
            res: &Res,
            lay: Lay,
            rows: usize,
            act: &GemmAct32,
            route: &GemmRoute,
            input: GemmInput,
            y: &mut DeviceBuffer<f32>,
            n_slots: usize,
        ) -> Result<Vec<f32>, GateError> {
            let stream = dev.gpu.stream();
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            dev.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: res.weight(lay)?,
                    rows_per_expert: rows,
                    act,
                    route,
                    input,
                    y,
                },
            )?;
            stream.synchronize()?;
            let mut out = y.to_host_vec(stream)?;
            out.truncate(n_slots * rows);
            Ok(out)
        }

        /// One NVFP4 GEMM over the planes and the experts' scales of a
        /// [`WtsNv`] stack, `y` set to [`SENT`] first; `y`'s first
        /// `n_slots·rows` values back.
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn gemm_nv(
            dev: &Dev<'_>,
            planes: Nvfp4Stack<'_>,
            rows: usize,
            act: &GemmAct32,
            route: &GemmRoute,
            input: GemmInput,
            y: &mut DeviceBuffer<f32>,
            n_slots: usize,
        ) -> Result<Vec<f32>, GateError> {
            let stream = dev.gpu.stream();
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            dev.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Nvfp4Planes(planes),
                    rows_per_expert: rows,
                    act,
                    route,
                    input,
                    y,
                },
            )?;
            stream.synchronize()?;
            let mut out = y.to_host_vec(stream)?;
            out.truncate(n_slots * rows);
            Ok(out)
        }

        /// The quantizer against the host transcription, its NaN refusal,
        /// and the SwiGLU quantizer against its two-launch composition.
        fn quant_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_quant") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let mut ok = true;
            gpu.clear_fault()?;
            for k in KS {
                let mut act = GemmAct32::new(stream, 4096, k)?;
                for n in [1usize, 9, 4096] {
                    let x = activations(k, n, 9100 + (k + n) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32.enqueue_quantize_gemm32(
                        stream,
                        &xd,
                        n,
                        &mut act,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let (_, want) = host_act(&x, k, n);
                    let same = planes_equal(&act, n, &want, stream)?;
                    let fault = gpu.take_fault()?;
                    let pass = same && fault.is_none();
                    println!(
                        "gemm32 case=g32_quant K={k} cols={n} planes_eq_host={same} fault={} {}",
                        fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // A NaN and an infinity: each block refused, the named site raised.
            for (what, bad) in [("nan", f32::NAN), ("inf", f32::INFINITY)] {
                let (k, n) = (320usize, 9usize);
                let mut act = GemmAct32::new(stream, n, k)?;
                let mut x = activations(k, n, 9200);
                x[k + 2 * 32 + 5] = bad;
                let xd = DeviceBuffer::from_host(stream, &x)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, n, &mut act, gpu.unlabelled_sink())?;
                stream.synchronize()?;
                let (_, want) = host_act(&x, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let d = act.d().to_host_vec(stream)?;
                let refused = d[act.steps() * 2 + 2].is_nan();
                let fault = gpu.take_fault()?;
                let site = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let pass = same && refused && site;
                println!(
                    "gemm32 case=g32_quant fault={what} K={k} planes_eq_host(block refused: NaN d, \
                     zero codes and sum)={same} d_nan={refused} site_quant_column={site} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // SwiGLU: the fused launch against `enqueue_swiglu` then the
            // quantizer, bit for bit, and its fault.
            for k in [96usize, 640] {
                let max = 4096;
                let mut fused = GemmAct32::new(stream, max, k)?;
                let mut split = GemmAct32::new(stream, max, k)?;
                let mut h = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
                for (n, nan) in [(1usize, false), (9, false), (4096, false), (9, true)] {
                    let gh = activations(k, n, 9300 + (k + n) as u32);
                    let mut gh = gh.into_iter().map(|v| 4.0 * v).collect::<Vec<_>>();
                    if nan {
                        gh[3 * k + 33] = f32::NAN;
                    }
                    let uh = activations(k, n, 9400 + (k + n) as u32);
                    let (g, u) = (
                        DeviceBuffer::from_host(stream, &gh)?,
                        DeviceBuffer::from_host(stream, &uh)?,
                    );
                    dev.g32.enqueue_swiglu_quant32(
                        stream,
                        &g,
                        &u,
                        n,
                        &mut fused,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let f_fault = gpu.take_fault()?;
                    gpu.elem().enqueue_swiglu(stream, &g, &u, n * k, &mut h)?;
                    dev.g32.enqueue_quantize_gemm32(
                        stream,
                        &h,
                        n,
                        &mut split,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let s_fault = gpu.take_fault()?;
                    let st = fused.steps();
                    let same = fused.q().to_host_vec(stream)?[..n * 16 * st]
                        == split.q().to_host_vec(stream)?[..n * 16 * st]
                        && nan_bits_equal(
                            &fused.d().to_host_vec(stream)?[..n * 2 * st],
                            &split.d().to_host_vec(stream)?[..n * 2 * st],
                        )
                        && fused.s().to_host_vec(stream)?[..n * 2 * st]
                            == split.s().to_host_vec(stream)?[..n * 2 * st];
                    let hh = h.to_host_vec(stream)?;
                    let (_, host) = host_act(&hh[..n * k], k, n);
                    let host_ok = planes_equal(&fused, n, &host, stream)?;
                    let want = nan.then(|| Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                    let pass = same && host_ok && f_fault == want && s_fault == want;
                    println!(
                        "gemm32 case=g32_quant swiglu K={k} cols={n} nan={nan} planes_eq_composition={same} \
                         planes_eq_host_of_swiglu_rows={host_ok} fault={} {}",
                        f_fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // The card-slots-only SwiGLU (`swiglu_quant32_sel`) over a
            // remapped route's places: a first plain launch seeds every
            // column's bytes, then the `_sel` launch over other rows — the
            // card columns the plain launch's bits over those rows, the host
            // columns the seed's, untouched; a NaN in a host column raising
            // nothing; a place in [n_card, HOST) `ExpertId` with nothing
            // written.
            {
                let (k, n, n_card) = (640usize, 33usize, 20usize);
                let place = |c: usize| {
                    if c.is_multiple_of(3) {
                        HOST
                    } else {
                        ((c * 7) % n_card) as u32
                    }
                };
                let sel: Vec<u32> = (0..n).map(place).collect();
                let sel_d = DeviceBuffer::from_host(stream, &sel)?;
                let mut plain = GemmAct32::new(stream, n, k)?;
                let mut sela = GemmAct32::new(stream, n, k)?;
                let seed = |s: u32| -> Vec<f32> {
                    activations(k, n, s).into_iter().map(|v| 4.0 * v).collect()
                };
                let (g0, u0, g1, u1) = (
                    DeviceBuffer::from_host(stream, &seed(9600))?,
                    DeviceBuffer::from_host(stream, &activations(k, n, 9601))?,
                    DeviceBuffer::from_host(stream, &seed(9602))?,
                    DeviceBuffer::from_host(stream, &activations(k, n, 9603))?,
                );
                dev.g32.enqueue_swiglu_quant32(
                    stream,
                    &g0,
                    &u0,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let steps = sela.steps();
                let row = |a: &GemmAct32| -> Result<Planes32, GateError> {
                    Ok((
                        a.q().to_host_vec(stream)?,
                        a.d().to_host_vec(stream)?,
                        a.s().to_host_vec(stream)?,
                    ))
                };
                let before = row(&sela)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &g1,
                    &u1,
                    &sel_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                dev.g32.enqueue_swiglu_quant32(
                    stream,
                    &g1,
                    &u1,
                    n,
                    &mut plain,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let (got, want) = (row(&sela)?, row(&plain)?);
                let eq = |a: &[u32], b: &[u32], c: usize| {
                    a[c * 16 * steps..][..16 * steps] == b[c * 16 * steps..][..16 * steps]
                };
                let mut card = 0usize;
                let mut host_ok = true;
                let mut card_ok = true;
                for (c, &sel_c) in sel.iter().enumerate().take(n) {
                    let same = eq(&got.0, &want.0, c)
                        && got.1[c * 2 * steps..][..2 * steps]
                            == want.1[c * 2 * steps..][..2 * steps]
                        && got.2[c * 2 * steps..][..2 * steps]
                            == want.2[c * 2 * steps..][..2 * steps];
                    if sel_c == HOST {
                        let kept = eq(&got.0, &before.0, c)
                            && got.1[c * 2 * steps..][..2 * steps]
                                == before.1[c * 2 * steps..][..2 * steps]
                            && got.2[c * 2 * steps..][..2 * steps]
                                == before.2[c * 2 * steps..][..2 * steps];
                        host_ok &= kept;
                    } else {
                        card += 1;
                        card_ok &= same;
                    }
                }
                let pass = card_ok && host_ok && fault.is_none();
                println!(
                    "gemm32 case=g32_quant swiglu_sel K={k} cols={n} card_cols_eq_plain={card_ok} \
                     ({card} of them) host_cols_untouched={host_ok} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
                // A NaN in a host column is never read: no fault, the host
                // column still the seed's, the card columns still written.
                let mut gn = seed(9604);
                let host_col = (0..n).find(|&c| sel[c] == HOST).unwrap_or(0);
                gn[host_col * k + 5] = f32::NAN;
                let gn = DeviceBuffer::from_host(stream, &gn)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &gn,
                    &u1,
                    &sel_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let got = row(&sela)?;
                let host_kept = eq(&got.0, &before.0, host_col);
                let pass = fault.is_none() && host_kept;
                println!(
                    "gemm32 case=g32_quant swiglu_sel_nan_host K={k} col={host_col} \
                     host_col_untouched={host_kept} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
                // A place in [n_card, HOST): `ExpertId`, its column unwritten.
                let mut stray = sel;
                stray[host_col] = n_card as u32;
                let stray_d = DeviceBuffer::from_host(stream, &stray)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &g1,
                    &u1,
                    &stray_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let got = row(&sela)?;
                let unwritten = eq(&got.0, &before.0, host_col);
                let pass = fault == Some(Fault::at(LAYER_NONE, FaultSite::ExpertId)) && unwritten;
                println!(
                    "gemm32 case=g32_quant swiglu_sel_stray K={k} col={host_col} \
                     col_unwritten={unwritten} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The rule-taking SwiGLU quantizer (`swiglu_act_quant32`): under ik's
        /// clamp, bit for bit the CPU tier's ik-verified `qdot::swiglu_clamp`
        /// then the host quantizer, each side of the clamp met; under the
        /// plain rule, `swiglu_quant32`'s bytes; a non-finite input refused.
        fn swiglu_act_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_swiglu_act") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            // GLM-5.3's dense lead and shared expert clamp at the file's
            // `swiglu_clamp_shexp` (`limit_shexp` of
            // `crates/model/src/arch/glm5next/hparams.rs`), its routed experts
            // at `swiglu_clamp_exp`; 10 stands for them here, 0.5 clamps most
            // values.
            const LIMITS: [f32; 2] = [10.0, 0.5];
            let max = 512;
            // Per limit: values whose silu(g) passes it, whose u passes +limit
            // or -limit, and that neither clamp touches.
            let mut sides = [[0usize; 4]; 2];
            // K 96 ends in a one-block step; 2048 and 12288 are the down's K of
            // GLM's shared expert and dense lead.
            for k in [96usize, 2048, 12288] {
                let mut act = GemmAct32::new(stream, max, k)?;
                let mut plain = GemmAct32::new(stream, max, k)?;
                for n in [1usize, 9, max] {
                    let scaled = |seed: u32| -> Vec<f32> {
                        activations(k, n, seed)
                            .into_iter()
                            .map(|v| 12.0 * v)
                            .collect()
                    };
                    let (gh, uh) = (scaled(9700 + (k + n) as u32), scaled(9800 + (k + n) as u32));
                    let (g, u) = (
                        DeviceBuffer::from_host(stream, &gh)?,
                        DeviceBuffer::from_host(stream, &uh)?,
                    );
                    for (li, limit) in LIMITS.into_iter().enumerate() {
                        dev.g32.enqueue_swiglu_act_quant32(
                            stream,
                            &g,
                            &u,
                            Act::SwigluClamp { limit },
                            n,
                            &mut act,
                            sink,
                        )?;
                        stream.synchronize()?;
                        let fault = gpu.take_fault()?;
                        let mut h = vec![0.0f32; n * k];
                        qdot::swiglu_clamp(&gh, &uh, limit, &mut h);
                        let (_, want) = host_act(&h, k, n);
                        let same = planes_equal(&act, n, &want, stream)?;
                        for (&gv, &uv) in gh.iter().zip(&uh) {
                            let over = silu_ik(gv) > limit;
                            let c = &mut sides[li];
                            c[0] += usize::from(over);
                            c[1] += usize::from(uv > limit);
                            c[2] += usize::from(uv < -limit);
                            c[3] += usize::from(!over && uv.abs() <= limit);
                        }
                        let pass = same && fault.is_none();
                        println!(
                            "gemm32 case=g32_swiglu_act clamp K={k} cols={n} limit={limit} \
                             planes_eq_host_of_ik_clamp={same} fault={} {}",
                            fault.map_or_else(|| "none".into(), |f| f.to_string()),
                            verdict(pass)
                        );
                        ok &= pass;
                    }
                    // The plain rule: the bytes of `swiglu_quant32`.
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &g,
                        &u,
                        Act::SiluMul,
                        n,
                        &mut act,
                        sink,
                    )?;
                    dev.g32
                        .enqueue_swiglu_quant32(stream, &g, &u, n, &mut plain, sink)?;
                    stream.synchronize()?;
                    let fault = gpu.take_fault()?;
                    let st = plain.steps();
                    let want: Planes32 = (
                        plain.q().to_host_vec(stream)?[..n * 16 * st].to_vec(),
                        plain.d().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                        plain.s().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                    );
                    let same = planes_equal(&act, n, &want, stream)?;
                    let pass = same && fault.is_none();
                    println!(
                        "gemm32 case=g32_swiglu_act silu_mul K={k} cols={n} \
                         planes_eq_swiglu_quant32={same} fault={} {}",
                        fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            for (li, limit) in LIMITS.into_iter().enumerate() {
                let [sg, up, dn, free] = sides[li];
                let pass = sg > 0 && up > 0 && dn > 0 && free > 0;
                println!(
                    "gemm32 case=g32_swiglu_act sides limit={limit} silu_over={sg} up_over={up} \
                     up_under={dn} unclamped={free} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // The OAI rule through the same quantizer: bit for bit the CPU twin
            // `qdot::swiglu_oai` then the host quantizer, the limit biting on
            // some values and not on others.
            let (k, n) = (2048usize, 9usize);
            let mut act = GemmAct32::new(stream, n, k)?;
            let scaled = |seed: u32| -> Vec<f32> {
                activations(k, n, seed)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect()
            };
            let (gh, uh) = (scaled(9700 + (k + n) as u32), scaled(9800 + (k + n) as u32));
            let (g, u) = (
                DeviceBuffer::from_host(stream, &gh)?,
                DeviceBuffer::from_host(stream, &uh)?,
            );
            for limit in [7.0f32, 0.5] {
                let rule = swiglu_host::oai(limit);
                dev.g32
                    .enqueue_swiglu_act_quant32(stream, &g, &u, rule, n, &mut act, sink)?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let (h, bitten) = swiglu_host::combine(rule, &gh, &uh).expect("a limited rule");
                let (_, want) = host_act(&h, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let pass = same && fault.is_none() && bitten > 0 && bitten < n * k;
                println!(
                    "gemm32 case=g32_swiglu_act swiglu_oai K={k} cols={n} limit={limit} \
                     bitten={bitten}/{} planes_eq_host_of_oai={same} fault={} {}",
                    n * k,
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            // A non-finite input refuses its block under the clamp, where the
            // rule alone would carry a NaN or infinite `u`, or an infinite `g`,
            // to a finite value: the host side is the clamp's rows with that
            // value NaN.
            let (k, n) = (2048usize, 9usize);
            let mut act = GemmAct32::new(stream, n, k)?;
            let at = 3 * k + 33;
            let blk = 3 * act.steps() * 2 + 1;
            let limit = LIMITS[0];
            for (what, on_g, bad) in [
                ("g_nan", true, f32::NAN),
                ("g_inf", true, f32::INFINITY),
                ("u_nan", false, f32::NAN),
                ("u_inf", false, f32::NEG_INFINITY),
            ] {
                let mut gh: Vec<f32> = activations(k, n, 9900)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect();
                let mut uh: Vec<f32> = activations(k, n, 9901)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect();
                if on_g {
                    gh[at] = bad;
                } else {
                    uh[at] = bad;
                }
                let (g, u) = (
                    DeviceBuffer::from_host(stream, &gh)?,
                    DeviceBuffer::from_host(stream, &uh)?,
                );
                dev.g32.enqueue_swiglu_act_quant32(
                    stream,
                    &g,
                    &u,
                    Act::SwigluClamp { limit },
                    n,
                    &mut act,
                    sink,
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let mut h = vec![0.0f32; n * k];
                qdot::swiglu_clamp(&gh, &uh, limit, &mut h);
                h[at] = f32::NAN;
                let (_, want) = host_act(&h, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let refused = act.d().to_host_vec(stream)?[blk].is_nan();
                let site = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let pass = same && refused && site;
                println!(
                    "gemm32 case=g32_swiglu_act fault={what} K={k} limit={limit} \
                     planes_eq_host(block refused)={same} d_nan={refused} \
                     site_quant_column={site} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The squared-ReLU quantizer (`relu_sqr_quant32`): bit for bit the host
        /// quantizer of the rule and the two launches it replaces
        /// (`enqueue_relu_sqr`, `quantize_gemm32`); a non-finite input, or a
        /// finite one whose square overflows, refused and raised.
        fn relu_sqr_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_relu_sqr") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            // The rule on the host: a non-finite value stays as it is.
            let rule = |v: f32| -> f32 {
                if !v.is_finite() {
                    v
                } else {
                    let r = v.max(0.0);
                    r * r
                }
            };
            let max = 512;
            // K 96 ends in a one-block step; 1856 and 3712 are the down's K
            // of Lightning's routed and shared experts.
            for k in [96usize, 1856, 3712] {
                let mut act = GemmAct32::new(stream, max, k)?;
                let mut two = GemmAct32::new(stream, max, k)?;
                for n in [1usize, 9, max] {
                    let xh: Vec<f32> = activations(k, n, 9950 + (k + n) as u32)
                        .into_iter()
                        .map(|v| 3.0 * v)
                        .collect();
                    let x = DeviceBuffer::from_host(stream, &xh)?;
                    dev.g32
                        .enqueue_quantize_gemm32_relu_sqr(stream, &x, n, &mut act, sink)?;
                    let mut sq = DeviceBuffer::<f32>::zeroed(stream, n * k)?;
                    gpu.elem().enqueue_relu_sqr(stream, &x, n * k, &mut sq)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &sq, n, &mut two, sink)?;
                    stream.synchronize()?;
                    let fault = gpu.take_fault()?;
                    let want_h: Vec<f32> = xh.iter().map(|&v| rule(v)).collect();
                    let (_, want) = host_act(&want_h, k, n);
                    let same_host = planes_equal(&act, n, &want, stream)?;
                    let st = two.steps();
                    let want2: Planes32 = (
                        two.q().to_host_vec(stream)?[..n * 16 * st].to_vec(),
                        two.d().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                        two.s().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                    );
                    let same_two = planes_equal(&act, n, &want2, stream)?;
                    let (neg, pos) = (
                        xh.iter().filter(|v| **v < 0.0).count(),
                        xh.iter().filter(|v| **v > 0.0).count(),
                    );
                    let pass = same_host && same_two && fault.is_none() && neg > 0 && pos > 0;
                    println!(
                        "gemm32 case=g32_relu_sqr K={k} cols={n} planes_eq_host_rule={same_host} \
                         planes_eq_relu_sqr_then_quantize_gemm32={same_two} negative={neg} \
                         positive={pos} fault={} {}",
                        fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // A value the rule cannot carry refuses its block: the host side is
            // the rule's rows, the value itself (or its overflowed square)
            // non-finite there.
            let (k, n) = (1856usize, 9usize);
            let mut act = GemmAct32::new(stream, n, k)?;
            let at = 3 * k + 33;
            let blk = 3 * act.steps() * 2 + 1;
            for (what, bad) in [
                ("nan", f32::NAN),
                ("pos_inf", f32::INFINITY),
                ("neg_inf", f32::NEG_INFINITY),
                ("overflow", 1.0e20),
            ] {
                let mut xh: Vec<f32> = activations(k, n, 9960)
                    .into_iter()
                    .map(|v| 3.0 * v)
                    .collect();
                xh[at] = bad;
                let x = DeviceBuffer::from_host(stream, &xh)?;
                dev.g32
                    .enqueue_quantize_gemm32_relu_sqr(stream, &x, n, &mut act, sink)?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let want_h: Vec<f32> = xh.iter().map(|&v| rule(v)).collect();
                let (_, want) = host_act(&want_h, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let refused = act.d().to_host_vec(stream)?[blk].is_nan();
                let site = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let pass = same && refused && site;
                println!(
                    "gemm32 case=g32_relu_sqr fault={what} K={k} \
                     planes_eq_host(block refused)={same} d_nan={refused} \
                     site_quant_column={site} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The dense cases: per K a Q8_0 stack (planes and file words from
        /// the same bytes) and a Q5_1 stack of [`ROWS`] rows, per column
        /// count one quantization and the three entries through the dense
        /// table; each against the reference, the two Q8_0 layouts equal bit
        /// for bit; then one captured route and GEMM replayed.
        fn dense_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_dense") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let max = MS[MS.len() - 1];
            let mut route = GemmRoute::new(stream, max, 1)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max * ROWS)?;
            let mut ok = true;
            for (ki, k) in KS.into_iter().enumerate() {
                let (w8, dec8) = weights(false, ROWS, k, 0x3280 + ki as u64)?;
                let (w5, dec5) = weights(true, ROWS, k, 0x3250 + ki as u64)?;
                let r8 = Res::new(dev, &w8, false, ROWS)?;
                let r5 = Res::new(dev, &w5, true, ROWS)?;
                let mut act = GemmAct32::new(stream, max, k)?;
                for m in MS {
                    let x = activations(k, m, 9500 + (k * 7 + m) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    stream.synchronize()?;
                    let (ha, want) = host_act(&x, k, m);
                    let act_ok = planes_equal(&act, m, &want, stream)?;
                    let mut y8 = Vec::new();
                    for (lay, w, res, dec) in [
                        (Lay::Q8Plane, &w8, &r8, dec8),
                        (Lay::Q8File, &w8, &r8, dec8),
                        (Lay::Q51, &w5, &r5, dec5),
                    ] {
                        let got = gemm(
                            dev,
                            res,
                            lay,
                            ROWS,
                            &act,
                            &route,
                            GemmInput::PerSlot,
                            &mut y,
                            m,
                        )?;
                        let unwritten =
                            got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                        let Checked {
                            checked,
                            band_off,
                            bits_off,
                            worst,
                            first,
                            ..
                        } = check(w, lay == Lay::Q51, ROWS, &ha, &|_| Some(0), &|s| s, &got, m)?;
                        let layouts = match lay {
                            Lay::Q8Plane => {
                                y8 = got.clone();
                                "-".to_string()
                            }
                            Lay::Q8File => bits_equal(&got, &y8).to_string(),
                            Lay::Iq4Nl | Lay::Q51 => "-".to_string(),
                        };
                        let fault = gpu.take_fault()?;
                        let pass = act_ok
                            && dec
                            && unwritten == 0
                            && band_off == 0
                            && bits_off == 0
                            && layouts != "false"
                            && fault.is_none();
                        println!(
                            "gemm32 case=g32_dense lay={lay:?} K={k} m={m} rows={ROWS} act_eq_host={act_ok} \
                             decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                             worst_err_over_band={worst:.3} band_off={band_off} contract_bits_differ={bits_off} \
                             file_eq_plane={layouts} y_fnv={:016x} {}",
                            Fnv1a64::default().f32s(&got).value(),
                            verdict(pass)
                        );
                        if let Some(f) = first {
                            println!(
                                "gemm32 case=g32_dense lay={lay:?} K={k} m={m} first_failure: {f}"
                            );
                        }
                        ok &= pass;
                    }
                }
            }
            // Graph: the dense table and a plane GEMM captured, replayed.
            let (k, m) = (2560usize, 512usize);
            let (w8, _) = weights(false, ROWS, k, 0x32a0)?;
            let r8 = Res::new(dev, &w8, false, ROWS)?;
            let mut act = GemmAct32::new(stream, m, k)?;
            let xd = DeviceBuffer::from_host(stream, &activations(k, m, 9600))?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
            dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
            let eager = gemm(
                dev,
                &r8,
                Lay::Q8Plane,
                ROWS,
                &act,
                &route,
                GemmInput::PerSlot,
                &mut y,
                m,
            )?;
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            let w = r8.weight(Lay::Q8Plane)?;
            let (gk, g32, route_r, act_r, y_r) = (dev.gk, &dev.g32, &mut route, &act, &mut y);
            let graph = gpu.capture(|s| {
                gk.enqueue_route_dense(s, m, route_r, sink)?;
                g32.enqueue_gemm32(
                    s,
                    Gemm32Args {
                        w,
                        rows_per_expert: ROWS,
                        act: act_r,
                        route: route_r,
                        input: GemmInput::PerSlot,
                        y: y_r,
                    },
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let mut replay = y.to_host_vec(stream)?;
            replay.truncate(m * ROWS);
            let (same, nodes) = (bits_equal(&replay, &eager), graph.node_count());
            let pass = same && nodes == 2;
            println!(
                "gemm32 case=g32_dense graph K={k} m={m} graph_nodes={nodes} replay_eq_eager={same} {}",
                verdict(pass)
            );
            Ok(ok && pass)
        }

        /// The halfword a BF16 store check fills its output with first: a cell
        /// still holding it past the product was not written.
        const GUARD: u16 = 0xAAAA;

        /// The grouped GEMM's BF16 store against its F32 store of the same
        /// launch rounded to nearest even, the guard past the cells untouched:
        /// a synthetic stack of each type a dense site reads over the dense
        /// table at a ragged and a multi-tile column count, and a type no
        /// dense site reads refused by name with the output untouched.
        fn bf16_store_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("bf16_store") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (rows, k) = (320usize, 2048usize);
            let ms = [7usize, 300];
            let max = ms[ms.len() - 1];
            let mut act = GemmAct::new(stream, max, k)?;
            let mut route = GemmRoute::new(stream, max, 1)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
            let mut ok = true;
            for (i, ty) in [
                GemmWeight::Q3K,
                GemmWeight::Q4K,
                GemmWeight::Q5K,
                GemmWeight::Q6K,
                GemmWeight::Iq4Xs,
                GemmWeight::Iq3Xxs,
            ]
            .into_iter()
            .enumerate()
            {
                let words = bytes_to_words(&synthetic(ty, rows * k / 256, 0xbf16 + i as u64));
                let w = DeviceTensor::upload(stream, &words, rows, words.len() / rows)?;
                let dense = ty != GemmWeight::Iq3Xxs;
                for m in ms {
                    let x = DeviceBuffer::from_host(stream, &activations(k, m, 9900 + m as u32))?;
                    gpu.enqueue_quantize_gemm(&x, m, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    macro_rules! args {
                        ($y:expr) => {
                            GemmArgs {
                                ty,
                                w: &w,
                                rows_per_expert: rows,
                                act: &act,
                                route: &route,
                                input: GemmInput::PerSlot,
                                y: $y,
                            }
                        };
                    }
                    let mut yb = DeviceBuffer::from_host(stream, &vec![GUARD; m * rows + 16])?;
                    let got = dev.gk.enqueue_gemm_bf16(stream, args!(&mut yb));
                    let yb = yb.to_host_vec(stream)?;
                    let (cells, guard) = yb.split_at(m * rows);
                    let pass = guard.iter().all(|&b| b == GUARD)
                        && if dense {
                            got.is_ok() && {
                                dev.gk.enqueue_gemm(stream, args!(&mut y))?;
                                let f = y.to_host_vec(stream)?;
                                cells
                                    .iter()
                                    .zip(&f[..m * rows])
                                    .all(|(&b, &v)| b == cuda_device::convert::f32_to_bf16_rne(v))
                            }
                        } else {
                            cells.iter().all(|&b| b == GUARD)
                                && matches!(
                                    &got,
                                    Err(GpuError::Shape { what, .. })
                                        if *what == "GemmKernels::enqueue_gemm_bf16"
                                )
                        }
                        && gpu.take_fault()?.is_none();
                    println!(
                        "gemm32 case=bf16_store ty={ty:?} m={m} rows={rows} K={k} dense_site={dense} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// The BF16 store of the product `f32` of `w` over `act` and the
        /// table `route`: each cell `f32` rounded to nearest even
        /// and the guard past the cells untouched.
        fn bf16_product32(
            dev: &Dev<'_>,
            w: Gemm32Weight<'_>,
            rows: usize,
            (act, route): (&GemmAct32, &GemmRoute),
            f32: &[f32],
        ) -> Result<bool, GateError> {
            let stream = dev.gpu.stream();
            let mut yb = DeviceBuffer::from_host(stream, &vec![GUARD; f32.len() + 16])?;
            dev.g32.enqueue_gemm32_bf16(
                stream,
                Gemm32Args {
                    w,
                    rows_per_expert: rows,
                    act,
                    route,
                    input: GemmInput::PerSlot,
                    y: &mut yb,
                },
            )?;
            let yb = yb.to_host_vec(stream)?;
            let (cells, guard) = yb.split_at(f32.len());
            Ok(guard.iter().all(|&b| b == GUARD)
                && cells
                    .iter()
                    .zip(f32)
                    .all(|(&b, &v)| b == cuda_device::convert::f32_to_bf16_rne(v)))
        }

        /// Both Q8_0-plane entries over the dense table on one stack of
        /// 4112 rows at K 2560 and K 2592 (11.2 and 11.3 MB of planes, past
        /// `GEMM32_RASTER_BYTES`): 16, 65 and 1100 columns, each against the
        /// reference and the contract's transcription (all outputs at 16/65
        /// columns, a rotating quarter at 1100). Captured nodes name the launched
        /// entry; a separate grid below the loaded wave checks the selector.
        fn slab_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_slab") {
                return Ok(true);
            }
            /// Rows of the stack: 32 full 128-row slabs and one of 16.
            const SLAB_ROWS: usize = 4112;
            let ms = [16usize, 65, 1100];
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let max = ms[ms.len() - 1];
            let mut route = GemmRoute::new(stream, max, 1)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max * SLAB_ROWS)?;
            let wave = dev.g32.q8_0p_wave();
            let selector_ok = !dev.g32.q8_0p_slab(wave - 1, 65, 1)
                && !dev.g32.q8_0p_slab(u32::MAX, 64, 1)
                && dev.g32.q8_0p_slab(wave, 65, 1);
            println!(
                "gemm32 case=g32_slab_selector wave={wave} {}",
                verdict(selector_ok)
            );
            let mut ok = selector_ok;
            // K 2592 is 81 blocks a row, so most rows start inside a group
            // of eight scales and read the window `@prep` binds before the
            // window first moves; K 2560 starts every row on a group.
            for k in [2560usize, 2592] {
                let (w, dec) = weights(false, SLAB_ROWS, k, 0x3510)?;
                let res = Res::new(dev, &w, false, SLAB_ROWS)?;
                let mut act = GemmAct32::new(stream, max, k)?;
                for m in ms {
                    let x = activations(k, m, 9700 + m as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    stream.synchronize()?;
                    let (ha, want) = host_act(&x, k, m);
                    let act_ok = planes_equal(&act, m, &want, stream)?;
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    let weight = res.weight(Lay::Q8Plane)?;
                    let graph = gpu.capture(|stream| {
                        dev.g32.enqueue_gemm32(
                            stream,
                            Gemm32Args {
                                w: weight,
                                rows_per_expert: SLAB_ROWS,
                                act: &act,
                                route: &route,
                                input: GemmInput::PerSlot,
                                y: &mut y,
                            },
                        )
                    })?;
                    let nodes = graph.nodes()?;
                    let entry = match nodes.as_slice() {
                        [node] => node
                            .kernel
                            .as_ref()
                            .ok_or("Q8_0 GEMM node is not a kernel")?,
                        _ => return Err("Q8_0 GEMM capture must contain one node".into()),
                    };
                    // The launch's grid: 128-row slabs times the dense table's
                    // `m / 64 + 1` tiles (`gemm_max_tiles`).
                    let grid = u32::try_from(SLAB_ROWS.div_ceil(128) * (m / 64 + 1))?;
                    let expected_entry = if m > 64 && grid >= wave {
                        "gemm_q8_0p_slab"
                    } else {
                        "gemm_q8_0p"
                    };
                    let entry_ok = entry.name == expected_entry && entry.grid == [grid, 1, 1];
                    graph.launch(stream)?;
                    stream.synchronize()?;
                    let mut got = y.to_host_vec(stream)?;
                    got.truncate(m * SLAB_ROWS);
                    let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                    let Checked {
                        checked,
                        band_off,
                        bits_off,
                        worst,
                        first,
                        ..
                    } = check(&w, false, SLAB_ROWS, &ha, &|_| Some(0), &|s| s, &got, m)?;
                    let bf16 = bf16_product32(dev, weight, SLAB_ROWS, (&act, &route), &got)?;
                    let fault = gpu.take_fault()?;
                    let pass = act_ok
                        && dec
                        && bf16
                        // PIN(2026-10-20): captured entry and grid replace the recomputed selector assertion.
                        && entry_ok
                        && unwritten == 0
                        && band_off == 0
                        && bits_off == 0
                        && fault.is_none();
                    println!(
                        "gemm32 case=g32_slab K={k} m={m} rows={SLAB_ROWS} grid={grid} \
                         entry={} entry_ok={entry_ok} act_eq_host={act_ok} decode_vs_ggml={dec} bf16_eq_f32_rne={bf16} unwritten={unwritten} \
                         checked={checked} worst_err_over_band={worst:.3} band_off={band_off} \
                         contract_bits_differ={bits_off} y_fnv={:016x} {}",
                        entry.name,
                        Fnv1a64::default().f32s(&got).value(),
                        verdict(pass)
                    );
                    if let Some(f) = first {
                        println!("gemm32 case=g32_slab K={k} m={m} first_failure: {f}");
                    }
                    ok &= pass;
                }
                let (qs, d) = res.plane.as_ref().ok_or("a Q8_0 stack with no planes")?;
                let mut scales = d.buf().to_host_vec(stream)?;
                scales.insert(0, 0);
                let parent = DeviceBuffer::from_host(stream, &scales)?;
                let d_off = DeviceTensor::<u16>::window_of(&parent, 2, SLAB_ROWS, k / 32)?;
                let mut codes = qs.buf().to_host_vec(stream)?;
                codes.insert(0, 0);
                let code_parent = DeviceBuffer::from_host(stream, &codes)?;
                let qs_off = DeviceTensor::<u32>::window_of(&code_parent, 4, SLAB_ROWS, k / 4)?;
                // PIN(2026-10-20): per-plane checks replace the slab-only scale-alignment clause.
                for (name, qs, d, offset, m, expect_slab) in [
                    ("scale", qs, &*d_off, 2, max, true),
                    ("code", &*qs_off, d, 4, max, true),
                    ("code", &*qs_off, d, 4, ms[0], false),
                ] {
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    let got = dev.g32.enqueue_gemm32(
                        stream,
                        Gemm32Args {
                            w: Gemm32Weight::Q8_0Plane { qs, d },
                            rows_per_expert: SLAB_ROWS,
                            act: &act,
                            route: &route,
                            input: GemmInput::PerSlot,
                            y: &mut y,
                        },
                    );
                    let refused = matches!(
                        &got,
                        Err(bloomery_gpu::GpuError::Shape { what, detail })
                            if *what == "Gemm32Kernels::enqueue_gemm32"
                                && detail.contains(&format!("Q8_0 {name} plane"))
                                && detail.contains("is not 16-byte aligned")
                    );
                    let untouched = refused
                        && y.to_host_vec(stream)?
                            .iter()
                            .all(|v| v.to_bits() == SENT.to_bits());
                    let grid = u32::try_from(SLAB_ROWS.div_ceil(128) * (m / 64 + 1))?;
                    let slab = dev.g32.q8_0p_slab(grid, u32::try_from(m)?, 1);
                    let pass = slab == expect_slab && refused && untouched;
                    println!(
                        "gemm32 case=g32_slab_{name}_unaligned K={k} byte_offset={offset} m={m} slab={slab} \
                         named_shape={refused} output_untouched={untouched} result={got:?} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// Routed stacks: a Q8_0 file stack and a Q5_1 one of 16 experts ×
        /// [`ROWS`] rows at K 640 (the routed down's K), each slot its own
        /// column, and one Q8_0 plane stack of 8 experts at K 2560 whose two
        /// slots a token share the token's column; plus a Q5_1 stack at the
        /// routed down's full shape, 2560 rows × K 640. A stack whose slots
        /// read their own column (a routed down's) stores BF16 too
        /// (`enqueue_gemm32_bf16`): the F32 cells rounded to nearest even.
        fn routed_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_routed") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            for (name, lay, n_exp, rows, k, top_k, shared, tokens) in [
                (
                    "q8_0f_16x272x640",
                    Lay::Q8File,
                    16usize,
                    ROWS,
                    640usize,
                    1usize,
                    false,
                    &MS[..],
                ),
                (
                    "q5_1_16x272x640",
                    Lay::Q51,
                    16,
                    ROWS,
                    640,
                    1,
                    false,
                    &MS[..],
                ),
                (
                    "q8_0p_8x272x2560_top2",
                    Lay::Q8Plane,
                    8,
                    ROWS,
                    2560,
                    2,
                    true,
                    &MS[..],
                ),
                (
                    "q5_1_8x2560x640",
                    Lay::Q51,
                    8,
                    2560,
                    640,
                    1,
                    false,
                    &[9usize, 512, 4096][..],
                ),
            ] {
                let q51 = lay == Lay::Q51;
                let n_rows = n_exp * rows;
                let (w, dec) = weights(q51, n_rows, k, 0x3300 + k as u64 + n_exp as u64)?;
                let res = Res::new(dev, &w, q51, n_rows)?;
                let max_t = tokens[tokens.len() - 1];
                let max_slots = max_t * top_k;
                let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
                let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut act = GemmAct32::new(stream, max_slots, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                for &t in tokens {
                    for r in [Routing::Uniform, Routing::SameTopk] {
                        let ids = route_ids(r, t, top_k, n_exp, 0x77 ^ (t as u64) << 4);
                        let n_slots = ids.len();
                        let n_cols = if shared { t } else { n_slots };
                        let x = activations(k, n_cols, 9700 + (t + k) as u32);
                        let xd = DeviceBuffer::from_host(stream, &x)?;
                        dev.g32
                            .enqueue_quantize_gemm32(stream, &xd, n_cols, &mut act, sink)?;
                        ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                        dev.gk
                            .enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                        let input = if shared {
                            GemmInput::Shared { top_k }
                        } else {
                            GemmInput::PerSlot
                        };
                        let got = gemm(dev, &res, lay, rows, &act, &route, input, &mut y, n_slots)?;
                        let (ha, _) = host_act(&x, k, n_cols);
                        let unwritten =
                            got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                        let Checked {
                            checked,
                            band_off,
                            bits_off,
                            worst,
                            first,
                            ..
                        } = check(
                            &w,
                            q51,
                            rows,
                            &ha,
                            &|s| Some(ids[s] as usize),
                            &|s| s / top_k,
                            &got,
                            n_slots,
                        )?;
                        // A routed down's BF16 store (one column a slot): the F32
                        // cells rounded to nearest even.
                        let bf16 = shared
                            || bf16_product32(dev, res.weight(lay)?, rows, (&act, &route), &got)?;
                        let fault = gpu.take_fault()?;
                        let pass = dec
                            && unwritten == 0
                            && band_off == 0
                            && bits_off == 0
                            && bf16
                            && fault.is_none();
                        println!(
                            "gemm32 case=g32_routed {name} T={t} routing={r:?} slots={n_slots} \
                             decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                             worst_err_over_band={worst:.3} band_off={band_off} contract_bits_differ={bits_off} \
                             bf16_eq_f32_rne={bf16} y_fnv={:016x} {}",
                            Fnv1a64::default().f32s(&got).value(),
                            verdict(pass)
                        );
                        if let Some(f) = first {
                            println!("gemm32 case=g32_routed {name} T={t} first_failure: {f}");
                        }
                        ok &= pass;
                    }
                }
            }
            Ok(ok)
        }

        /// (rows, K) per `g32_nvfp4_dense` run: Bolt's two NVFP4 projections'
        /// shapes — `ssm_in` 18560 × 4096, `ssm_out` 4096 × 8192 — and 128
        /// rows at both K.
        const NV_SHAPES: [(usize, usize); 4] =
            [(18560, 4096), (128, 4096), (128, 8192), (4096, 8192)];
        /// Columns (and, top_k 1, slots) per `g32_nvfp4_dense` and
        /// `g32_nvfp4_routed` run: one, one past a tile's eight, one tile,
        /// seven.
        const NV_MS: [usize; 4] = [1, 9, 64, 448];

        /// The resident NVFP4 planes (`mxfp4`'s `repack`, the stack the dense
        /// gemv reads) in `gemm_nvfp4p` over the dense table at
        /// [`NV_SHAPES`] × [`NV_MS`]: the decode ggml's, every checked output in its
        /// band `γ(2K/32 + 4) · Σ_b |scale·d_a·(d0·i0 + d1·i1)|` of the f64
        /// reference and bit for bit the contract's transcription
        /// (`gemm32::dot32_nv`); a rerun bit-identical, outputs whose F32
        /// walk overflows before the tensor scale stored finite, a captured
        /// table and GEMM replayed.
        fn nvfp4_dense_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_nvfp4_dense") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let max = NV_MS[NV_MS.len() - 1];
            let mut ok = true;
            for (si, (rows, k)) in NV_SHAPES.into_iter().enumerate() {
                let (w, dec) = weights_nvfp4(rows, k, 0x3600 + si as u64)?;
                let planes = repack::<Nvfp4>(&w.bytes, rows, k)?;
                let qs = DeviceTensor::upload(stream, &planes.qs, rows, k / 8)?;
                let e = DeviceTensor::upload(stream, &planes.e, rows, k / 64)?;
                let scales = DeviceTensor::upload(stream, &[w.scale], 1, 1)?;
                let nv = Nvfp4Stack {
                    qs: &qs,
                    e: &e,
                    scales: &scales,
                };
                let mut route = GemmRoute::new(stream, max, 1)?;
                let mut act = GemmAct32::new(stream, max, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
                let mut y2 = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
                for m in NV_MS {
                    let x = activations(k, m, 9900 + (k * 5 + m + si) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    let got = gemm_nv(dev, nv, rows, &act, &route, GemmInput::PerSlot, &mut y, m)?;
                    let rerun = bits_equal(
                        &got,
                        &gemm_nv(dev, nv, rows, &act, &route, GemmInput::PerSlot, &mut y2, m)?,
                    );
                    let (ha, want) = host_act(&x, k, m);
                    let act_ok = planes_equal(&act, m, &want, stream)?;
                    let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                    let Checked {
                        checked,
                        band_off,
                        bits_off,
                        worst,
                        first,
                        ..
                    } = check_dot(
                        2 * (k / 32) + 4,
                        rows,
                        &|_| Some(0),
                        &|s| s,
                        &|row, col| {
                            dot32_nv(
                                &w.q[row * k..(row + 1) * k],
                                &w.ds[row * k / 16..(row + 1) * k / 16],
                                w.scale,
                                &ha,
                                col,
                            )
                        },
                        &got,
                        m,
                    )?;
                    let fault = gpu.take_fault()?;
                    let pass = dec
                        && act_ok
                        && rerun
                        && unwritten == 0
                        && band_off == 0
                        && bits_off == 0
                        && fault.is_none();
                    println!(
                        "gemm32 case=g32_nvfp4_dense rows={rows} K={k} m={m} decode_vs_ggml={dec} \
                         act_eq_host={act_ok} unwritten={unwritten} checked={checked} \
                         worst_err_over_band={worst:.3} band_off={band_off} \
                         contract_bits_differ={bits_off} rerun_eq={rerun} y_fnv={:016x} {}",
                        Fnv1a64::default().f32s(&got).value(),
                        verdict(pass)
                    );
                    if let Some(f) = first {
                        println!(
                            "gemm32 case=g32_nvfp4_dense rows={rows} K={k} m={m} first_failure: {f}"
                        );
                    }
                    ok &= pass;
                }
            }
            // Range: K 64, activation codes 127 and tensor scale 2^-120; the
            // F32 walk overflows and the exact walk stores the value given.
            // `range`: every code 2 under sub-block scales 0.5 (UE4M3 0x38),
            // `d_a` 2^120, 8128. `cancel`: codes (12 ×16, 1, 0 ×15) then their
            // negation under sub-block scales (224, 2^-10) (UE4M3 0x7e, 0x01),
            // `d_a` (1 + 2^-23)·2^120; the second step's fused F64
            // multiply-add keeps the first step's rounding error, 2^-33.
            let (rows, k, m) = (128usize, 64usize, 1usize);
            let unit: Vec<i8> = [[12; 16].as_slice(), &[1], &[0; 15]].concat();
            let mut cancel = [0u8; 36];
            cancel[..4].copy_from_slice(&[0x7e, 0x01, 0x7e, 0x01]);
            cancel[4..12].fill(0x77);
            cancel[12] = 0x01;
            cancel[20..28].fill(0xff);
            cancel[28] = 0x09;
            let range_cases = [
                (
                    "range",
                    std::array::from_fn(|i| if i < 4 { 0x38 } else { 0x22 }),
                    vec![2; 64],
                    [0.5; 4],
                    2f32.powi(120),
                    8128f32,
                ),
                (
                    "cancel",
                    cancel,
                    [unit.clone(), unit.iter().map(|c| -c).collect()].concat(),
                    [224.0, 2f32.powi(-10), 224.0, 2f32.powi(-10)],
                    f32::from_bits(0x7b80_0001),
                    2f32.powi(-33),
                ),
            ];
            for (tag, block, codes, wsc, d_a, want) in range_cases {
                let planes = repack::<Nvfp4>(&block.repeat(rows * k / 64), rows, k)?;
                let qs = DeviceTensor::upload(stream, &planes.qs, rows, k / 8)?;
                let e = DeviceTensor::upload(stream, &planes.e, rows, k / 64)?;
                let scale = 2f32.powi(-120);
                let scales = DeviceTensor::upload(stream, &[scale], 1, 1)?;
                let nv = Nvfp4Stack {
                    qs: &qs,
                    e: &e,
                    scales: &scales,
                };
                let mut route = GemmRoute::new(stream, m, 1)?;
                let mut act = GemmAct32::new(stream, m, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, m * rows)?;
                let x = vec![127.0 * d_a; k * m];
                let xd = DeviceBuffer::from_host(stream, &x)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
                dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                let got = gemm_nv(dev, nv, rows, &act, &route, GemmInput::PerSlot, &mut y, m)?;
                let (ha, _) = host_act(&x, k, m);
                let act_ok = ha.codes.iter().all(|&c| c == 127) && ha.d.iter().all(|&d| d == d_a);
                let Checked {
                    band_off,
                    bits_off,
                    first,
                    ..
                } = check_dot(
                    2 * (k / 32) + 4,
                    rows,
                    &|_| Some(0),
                    &|s| s,
                    &|_, col| dot32_nv(&codes, &wsc, scale, &ha, col),
                    &got,
                    m,
                )?;
                let fault = gpu.take_fault()?;
                let stored = got.iter().all(|v| v.to_bits() == want.to_bits());
                let pass = act_ok && stored && band_off == 0 && bits_off == 0 && fault.is_none();
                println!(
                    "gemm32 case=g32_nvfp4_dense {tag} K={k} m={m} act_codes_127={act_ok} \
                     want={want:e} stored_eq={stored} band_off={band_off} contract_bits_differ={bits_off} {}",
                    verdict(pass)
                );
                if let Some(f) = first {
                    println!("gemm32 case=g32_nvfp4_dense {tag} first_failure: {f}");
                }
                ok &= pass;
            }
            // Graph: the dense table and the NVFP4 GEMM captured, replayed.
            let (rows, k, m) = (128usize, 4096usize, 64usize);
            let (w, _) = weights_nvfp4(rows, k, 0x36a0)?;
            let planes = repack::<Nvfp4>(&w.bytes, rows, k)?;
            let qs = DeviceTensor::upload(stream, &planes.qs, rows, k / 8)?;
            let e = DeviceTensor::upload(stream, &planes.e, rows, k / 64)?;
            let scales = DeviceTensor::upload(stream, &[w.scale], 1, 1)?;
            let wgt = Nvfp4Stack {
                qs: &qs,
                e: &e,
                scales: &scales,
            };
            let mut act = GemmAct32::new(stream, m, k)?;
            let mut route = GemmRoute::new(stream, m, 1)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, m * rows)?;
            let xd = DeviceBuffer::from_host(stream, &activations(k, m, 9950))?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
            dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
            let eager = gemm_nv(dev, wgt, rows, &act, &route, GemmInput::PerSlot, &mut y, m)?;
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            let (gk, g32, route_r, act_r, y_r) = (dev.gk, &dev.g32, &mut route, &act, &mut y);
            let graph = gpu.capture(|s| {
                gk.enqueue_route_dense(s, m, route_r, sink)?;
                g32.enqueue_gemm32(
                    s,
                    Gemm32Args {
                        w: Gemm32Weight::Nvfp4Planes(wgt),
                        rows_per_expert: rows,
                        act: act_r,
                        route: route_r,
                        input: GemmInput::PerSlot,
                        y: y_r,
                    },
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let mut replay = y.to_host_vec(stream)?;
            replay.truncate(m * rows);
            let (same, nodes) = (bits_equal(&replay, &eager), graph.node_count());
            let pass = same && nodes == 2;
            println!(
                "gemm32 case=g32_nvfp4_dense graph K={k} m={m} graph_nodes={nodes} \
                 replay_eq_eager={same} {}",
                verdict(pass)
            );
            Ok(ok && pass)
        }

        /// The NVFP4 planes under routed tables: a 16-expert stack of 272
        /// rows at K 4096, each slot its own column, and an 8-expert stack at
        /// K 8192 whose two slots a token share the token's column, each
        /// expert its own tensor scale; uniform and
        /// same-expert routings, each slot's rows its expert's contract bits
        /// and a rerun bit-identical; then an id past the stack raises
        /// `ExpertId`, its slots refused and NaN, the rest the clean run's
        /// bits.
        fn nvfp4_routed_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_nvfp4_routed") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            for (name, n_exp, k, top_k, tokens) in [
                ("nvfp4_16x272x4096", 16usize, 4096usize, 1usize, &NV_MS[..]),
                ("nvfp4_8x272x8192_top2", 8, 8192, 2, &[8usize, 32, 224][..]),
            ] {
                let (rows, n_rows) = (ROWS, n_exp * ROWS);
                let (w, dec) = weights_nvfp4(n_rows, k, 0x3700 + k as u64 + n_exp as u64)?;
                let planes = repack::<Nvfp4>(&w.bytes, n_rows, k)?;
                let qs = DeviceTensor::upload(stream, &planes.qs, n_rows, k / 8)?;
                let e = DeviceTensor::upload(stream, &planes.e, n_rows, k / 64)?;
                let es: Vec<f32> = (0..n_exp)
                    .map(|x| w.scale * (1.0 + x as f32 / 16.0))
                    .collect();
                let scales = DeviceTensor::upload(stream, &es, n_exp, 1)?;
                let nv = Nvfp4Stack {
                    qs: &qs,
                    e: &e,
                    scales: &scales,
                };
                let max_slots = tokens[tokens.len() - 1] * top_k;
                let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
                let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut act = GemmAct32::new(stream, max_slots, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                let mut y2 = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                for &t in tokens {
                    for r in [Routing::Uniform, Routing::SameTopk] {
                        let ids = route_ids(r, t, top_k, n_exp, 0x79 ^ (t + k) as u64);
                        let n_slots = ids.len();
                        let n_cols = if top_k == 1 { n_slots } else { t };
                        let x = activations(k, n_cols, 9970 + (t + k) as u32);
                        let xd = DeviceBuffer::from_host(stream, &x)?;
                        dev.g32
                            .enqueue_quantize_gemm32(stream, &xd, n_cols, &mut act, sink)?;
                        ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                        dev.gk
                            .enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                        let input = if top_k == 1 {
                            GemmInput::PerSlot
                        } else {
                            GemmInput::Shared { top_k }
                        };
                        let got = gemm_nv(dev, nv, rows, &act, &route, input, &mut y, n_slots)?;
                        let rerun = bits_equal(
                            &got,
                            &gemm_nv(dev, nv, rows, &act, &route, input, &mut y2, n_slots)?,
                        );
                        let (ha, _) = host_act(&x, k, n_cols);
                        let unwritten =
                            got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                        let Checked {
                            checked,
                            band_off,
                            bits_off,
                            worst,
                            first,
                            ..
                        } = check_dot(
                            2 * (k / 32) + 4,
                            rows,
                            &|s| Some(ids[s] as usize),
                            &|s| if top_k == 1 { s } else { s / top_k },
                            &|row, col| {
                                dot32_nv(
                                    &w.q[row * k..(row + 1) * k],
                                    &w.ds[row * k / 16..(row + 1) * k / 16],
                                    es[row / rows],
                                    &ha,
                                    col,
                                )
                            },
                            &got,
                            n_slots,
                        )?;
                        let fault = gpu.take_fault()?;
                        let pass = dec
                            && rerun
                            && unwritten == 0
                            && band_off == 0
                            && bits_off == 0
                            && fault.is_none();
                        println!(
                            "gemm32 case=g32_nvfp4_routed {name} T={t} routing={r:?} slots={n_slots} \
                             decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                             worst_err_over_band={worst:.3} band_off={band_off} \
                             contract_bits_differ={bits_off} rerun_eq={rerun} y_fnv={:016x} {}",
                            Fnv1a64::default().f32s(&got).value(),
                            verdict(pass)
                        );
                        if let Some(f) = first {
                            println!(
                                "gemm32 case=g32_nvfp4_routed {name} T={t} first_failure: {f}"
                            );
                        }
                        ok &= pass;
                    }
                }
                if top_k == 1 {
                    // An id past the stack: the route raises `ExpertId` and
                    // lists the slots refused; the GEMM writes their rows NaN,
                    // the rest the clean run's bits.
                    let t = NV_MS[2];
                    let mut ids = route_ids(Routing::Uniform, t, 1, n_exp, 0x69);
                    ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                    dev.gk.enqueue_route(stream, &ids_d, t, &mut route, sink)?;
                    let clean =
                        gemm_nv(dev, nv, rows, &act, &route, GemmInput::PerSlot, &mut y, t)?;
                    ids[5] = n_exp as u32;
                    ids[9] = u32::MAX;
                    ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                    dev.gk.enqueue_route(stream, &ids_d, t, &mut route, sink)?;
                    let got = gemm_nv(dev, nv, rows, &act, &route, GemmInput::PerSlot, &mut y, t)?;
                    let err = gpu
                        .take_fault()?
                        .map(|fault| GpuError::fault("gate_gemm", fault));
                    let site_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
                        if fault.site() == Some(FaultSite::ExpertId));
                    let refused_nan = [5usize, 9]
                        .iter()
                        .all(|&s| got[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
                    let others_clean = (0..t).filter(|s| *s != 5 && *s != 9).all(|s| {
                        bits_equal(
                            &got[s * rows..(s + 1) * rows],
                            &clean[s * rows..(s + 1) * rows],
                        )
                    });
                    let refused = route.refused_back(stream)?;
                    let pass = site_ok && refused_nan && others_clean && refused == [5, 9];
                    println!(
                        "gemm32 case=g32_nvfp4_routed {name} id_past_stack error=\"{}\" \
                         refused_slots_nan={refused_nan} others_eq_clean={others_clean} \
                         refused_listed={refused:?} {}",
                        err.as_ref()
                            .map_or_else(|| "none".to_string(), ToString::to_string),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// A SwiGLU pair's up GEMM as [`swiglu_sides`] launches it: a
        /// 32-value stack over the 32-value rows, or a grouped one over the
        /// q8_1 rows.
        #[derive(Clone, Copy)]
        enum UpW<'a> {
            G32(Gemm32Weight<'a>, &'a GemmAct32),
            Grouped(GemmWeight, &'a DeviceTensor<u32>, &'a GemmAct),
        }

        /// One SwiGLU epilogue comparison over the route's last fill: both
        /// activations seeded with the same planes, then the up GEMM and
        /// `swiglu_quant32` (`slots` all) or `swiglu_quant32_sel` (card
        /// slots) into `two` and the up launcher (`enqueue_up_swiglu32` of the
        /// up's family) into `one`, then the launcher captured into a graph
        /// and replayed into `one` over a fresh seed; each run's planes and
        /// fault, and the graph's node count.
        /// Each run's planes and fault (two-launch, fold, graph), and the graph's node count.
        type Sides = ([(Planes32, Option<Fault>); 3], usize);

        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn swiglu_sides(
            dev: &Dev<'_>,
            up: UpW<'_>,
            (rows, input): (usize, GemmInput),
            route: &GemmRoute,
            (g, slots): (&DeviceBuffer<f32>, SwigluSlots<'_>),
            u: &mut DeviceBuffer<f32>,
            seed: &DeviceBuffer<f32>,
            (two, one): (&mut GemmAct32, &mut GemmAct32),
        ) -> Result<Sides, GateError> {
            let (gpu, stream) = (dev.gpu, dev.gpu.stream());
            let sink = gpu.unlabelled_sink();
            let n_slots = route.filled().ok_or("gate_gemm: an unfilled route")?;
            let reseed = |a: &mut GemmAct32| {
                let cols = a.cols();
                dev.g32.enqueue_quantize_gemm32(stream, seed, cols, a, sink)
            };
            let fold = |s: &CudaStream, u: &mut DeviceBuffer<f32>, one: &mut GemmAct32| match up {
                UpW::G32(w, x) => dev.g32.enqueue_up_swiglu32(
                    s,
                    Gemm32Args {
                        w,
                        rows_per_expert: rows,
                        act: x,
                        route,
                        input,
                        y: u,
                    },
                    g,
                    slots,
                    one,
                    sink,
                ),
                UpW::Grouped(ty, w, x) => dev.gk.enqueue_up_swiglu32(
                    &dev.g32,
                    s,
                    GemmArgs {
                        ty,
                        w,
                        rows_per_expert: rows,
                        act: x,
                        route,
                        input,
                        y: u,
                    },
                    g,
                    slots,
                    one,
                    sink,
                ),
            };
            let planes = |a: &GemmAct32| -> Result<Planes32, GateError> {
                Ok((
                    a.q().to_host_vec(stream)?,
                    a.d().to_host_vec(stream)?,
                    a.s().to_host_vec(stream)?,
                ))
            };
            let poison = vec![f32::NAN; u.len()];
            let mut folded_u = DeviceBuffer::from_host(stream, &poison)?;
            reseed(two)?;
            reseed(one)?;
            match up {
                UpW::G32(w, x) => dev.g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w,
                        rows_per_expert: rows,
                        act: x,
                        route,
                        input,
                        y: &mut *u,
                    },
                )?,
                UpW::Grouped(ty, w, x) => dev.gk.enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty,
                        w,
                        rows_per_expert: rows,
                        act: x,
                        route,
                        input,
                        y: &mut *u,
                    },
                )?,
            }
            match slots {
                SwigluSlots::All => dev
                    .g32
                    .enqueue_swiglu_quant32(stream, g, u, n_slots, two, sink)?,
                SwigluSlots::Card { sel, n_card } => dev
                    .g32
                    .enqueue_swiglu_quant32_sel(stream, g, u, sel, n_card, n_slots, two, sink)?,
            }
            stream.synchronize()?;
            let f_two = gpu.take_fault()?;
            fold(stream, &mut folded_u, one)?;
            stream.synchronize()?;
            let f_one = gpu.take_fault()?;
            let eager = planes(one)?;
            reseed(one)?;
            folded_u.copy_from_host(stream, &poison)?;
            let graph = gpu.capture(|s| fold(s, &mut folded_u, one))?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let f_graph = gpu.take_fault()?;
            Ok((
                [
                    (planes(two)?, f_two),
                    (eager, f_one),
                    (planes(one)?, f_graph),
                ],
                graph.node_count(),
            ))
        }

        /// `enqueue_up_swiglu32`'s fold over NVFP4 planes against its two
        /// launches: module doc, `g32_nvfp4_swiglu`.
        fn nvfp4_swiglu_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_nvfp4_swiglu") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            let (n_map, top_k, k) = (24usize, 4usize, 512usize);
            let input = GemmInput::Shared { top_k };
            // Every third expert the host's, the others the card's places in
            // order.
            let map: Vec<u32> = (0..n_map)
                .map(|x| if x % 3 == 2 { HOST } else { (x - x / 3) as u32 })
                .collect();
            let n_card = n_map - n_map / 3;
            let map_d = DeviceBuffer::from_host(stream, &map)?;
            let max_t = NV_MS[NV_MS.len() - 1];
            let max_slots = max_t * top_k;
            let same =
                |a: &Planes32, b: &Planes32| a.0 == b.0 && nan_bits_equal(&a.1, &b.1) && a.2 == b.2;
            for rows in [640usize, 160] {
                let n_rows = n_card * rows;
                let (w, _) = weights_nvfp4(n_rows, k, 0x5100 + rows as u64)?;
                let planes = repack::<Nvfp4>(&w.bytes, n_rows, k)?;
                let qs = DeviceTensor::upload(stream, &planes.qs, n_rows, k / 8)?;
                let e = DeviceTensor::upload(stream, &planes.e, n_rows, k / 64)?;
                // Tensor scales near 2^-20 keep finite the outputs of token 0's
                // column, which lies far past the F32 walk's range.
                let es: Vec<f32> = (0..n_card)
                    .map(|x| (1.0 + x as f32 / 16.0) / 1_048_576.0)
                    .collect();
                let scales = DeviceTensor::upload(stream, &es, n_card, 1)?;
                let nv = Nvfp4Stack {
                    qs: &qs,
                    e: &e,
                    scales: &scales,
                };
                let mut route = GemmRoute::new(stream, max_slots, n_card)?;
                let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut sel_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut x = GemmAct32::new(stream, max_t, k)?;
                let mut u = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                let seed = DeviceBuffer::from_host(stream, &activations(rows, max_slots, 9810))?;
                let mut two = GemmAct32::new(stream, max_slots, rows)?;
                let mut one = GemmAct32::new(stream, max_slots, rows)?;
                for &t in &NV_MS {
                    let n_slots = t * top_k;
                    // From 64 tokens on, token 0's column sits far past the F32
                    // walk's range, so the tiles it lands in walk again exactly;
                    // its first slot is a card expert's.
                    let far = t >= 64;
                    let mut ids =
                        route_ids(Routing::Uniform, t, top_k, n_map, 0x5a ^ (t + rows) as u64);
                    if far {
                        ids[0] = (0..n_map as u32)
                            .find(|x| map[*x as usize] != HOST && !ids[1..top_k].contains(x))
                            .ok_or("gate_gemm: no card expert left for token 0")?;
                    }
                    let sel: Vec<u32> = ids.iter().map(|&i| map[i as usize]).collect();
                    let mut xs = activations(k, t, 9900 + t as u32);
                    if far {
                        xs[..k].iter_mut().for_each(|v| *v *= 1e35);
                    }
                    dev.g32.enqueue_quantize_gemm32(
                        stream,
                        &DeviceBuffer::from_host(stream, &xs)?,
                        t,
                        &mut x,
                        sink,
                    )?;
                    let gh = activations(rows, n_slots, 9700 + t as u32);
                    let mut run = |ids: &[u32], sel: &[u32], gh: &[f32]| {
                        ids_d.copy_from_host(stream, &pad(ids, max_slots))?;
                        sel_d.copy_from_host(stream, &pad(sel, max_slots))?;
                        dev.g32.enqueue_route_remap(
                            stream, &ids_d, &map_d, n_slots, &mut route, sink,
                        )?;
                        let g = DeviceBuffer::from_host(stream, gh)?;
                        let slots = SwigluSlots::Card {
                            sel: &sel_d,
                            n_card,
                        };
                        let sides = swiglu_sides(
                            dev,
                            UpW::G32(Gemm32Weight::Nvfp4Planes(nv), &x),
                            (rows, input),
                            &route,
                            (&g, slots),
                            &mut u,
                            &seed,
                            (&mut two, &mut one),
                        )?;
                        Ok::<_, GateError>((sides, u.to_host_vec(stream)?))
                    };
                    let (([(want, f_want), (got, f_got), (rep, f_rep)], nodes), uh) =
                        run(&ids, &sel, &gh)?;
                    let replay = nodes == 1 && same(&rep, &got) && f_rep == f_got;
                    // The up value of token 0's card slots against the F32
                    // range: past it, the tile's F32 walk overflowed and the
                    // stored value is the exact walk's.
                    let exact = (0..top_k).any(|s| {
                        sel[s] < n_card as u32
                            && uh[s * rows..(s + 1) * rows].iter().any(|&v| {
                                v.is_finite()
                                    && f64::from(v.abs()) / f64::from(es[sel[s] as usize])
                                        > f64::from(f32::MAX)
                            })
                    });
                    let pass = same(&want, &got)
                        && f_want.is_none()
                        && f_got.is_none()
                        && exact == far
                        && replay;
                    println!(
                        "gemm32 case=g32_nvfp4_swiglu rows={rows} K={k} T={t} slots={n_slots} \
                         planes_eq_two_launches={} exact_walk={exact} faults={:?}/{:?} \
                         replay_eq_eager={replay} act_fnv={:016x} {}",
                        same(&want, &got),
                        f_want,
                        f_got,
                        Fnv1a64::default().u32s(&got.0).value(),
                        verdict(pass)
                    );
                    ok &= pass;
                    if t != 64 {
                        continue;
                    }
                    // Faults, each case alone so either raise is seen
                    // missing: a NaN gate value in a card column; then a slot
                    // the route refuses on a card place (its up row NaN), one
                    // on a place no side serves and a listed slot whose place
                    // is in [n_card, HOST).
                    let card: Vec<usize> =
                        (0..n_slots).filter(|&s| sel[s] < n_card as u32).collect();
                    for nan_gate in [true, false] {
                        let (mut ids_f, mut sel_f, mut gh_f) =
                            (ids.clone(), sel.clone(), gh.clone());
                        let sites: &[FaultSite] = if nan_gate {
                            gh_f[card[0] * rows + 7] = f32::NAN;
                            &[FaultSite::QuantColumn]
                        } else {
                            ids_f[card[1]] = n_map as u32;
                            ids_f[card[2]] = n_map as u32 + 5;
                            sel_f[card[2]] = n_card as u32;
                            sel_f[card[3]] = n_card as u32 + 1;
                            &[FaultSite::ExpertId, FaultSite::QuantColumn]
                        };
                        let (([(want, f_want), (got, f_got), (rep, f_rep)], nodes), _) =
                            run(&ids_f, &sel_f, &gh_f)?;
                        let replay = nodes == 1 && same(&rep, &got) && f_rep == f_got;
                        let raised = Some(Fault::of_sites(LAYER_NONE, sites));
                        let pass =
                            same(&want, &got) && f_want == raised && f_got == raised && replay;
                        println!(
                            "gemm32 case=g32_nvfp4_swiglu rows={rows} T={t} nan_gate={nan_gate} \
                             planes_eq_two_launches={} fault_two={:?} fault_fold={:?} \
                             replay_eq_eager={replay} {}",
                            same(&want, &got),
                            f_want,
                            f_got,
                            verdict(pass)
                        );
                        ok &= pass;
                    }
                }
            }
            let (rows, k, n_exp) = (128usize, 64usize, 1usize);
            let (w, _) = weights_nvfp4(rows, k, 0x5180)?;
            let planes = repack::<Nvfp4>(&w.bytes, rows, k)?;
            let qs = DeviceTensor::upload(stream, &planes.qs, rows, k / 8)?;
            let e = DeviceTensor::upload(stream, &planes.e, rows, k / 64)?;
            let scales = DeviceTensor::upload(stream, &[w.scale], n_exp, 1)?;
            let nv = Nvfp4Stack {
                qs: &qs,
                e: &e,
                scales: &scales,
            };
            for n_slots in [1usize, 257] {
                let mut x = GemmAct32::new(stream, n_slots, k)?;
                dev.g32.enqueue_quantize_gemm32(
                    stream,
                    &DeviceBuffer::from_host(stream, &activations(k, n_slots, 9880))?,
                    n_slots,
                    &mut x,
                    sink,
                )?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                dev.g32.enqueue_route_remap(
                    stream,
                    &DeviceBuffer::from_host(stream, &vec![0; n_slots])?,
                    &DeviceBuffer::from_host(stream, &[HOST])?,
                    n_slots,
                    &mut route,
                    sink,
                )?;
                let empty_route =
                    route.read_back(stream)?.1.is_empty() && route.refused_back(stream)?.is_empty();
                let seed = DeviceBuffer::from_host(stream, &activations(rows, n_slots, 9881))?;
                let mut u = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut two = GemmAct32::new(stream, n_slots, rows)?;
                let mut one = GemmAct32::new(stream, n_slots, rows)?;
                for invalid_place in [false, true] {
                    let mut sel = vec![HOST; n_slots];
                    if invalid_place {
                        sel[n_slots - 1] = n_exp as u32;
                    }
                    let sel_d = DeviceBuffer::from_host(stream, &sel)?;
                    let slots = SwigluSlots::Card {
                        sel: &sel_d,
                        n_card: n_exp,
                    };
                    let ([(want, f_want), (got, f_got), (rep, f_rep)], nodes) = swiglu_sides(
                        dev,
                        UpW::G32(Gemm32Weight::Nvfp4Planes(nv), &x),
                        (rows, GemmInput::PerSlot),
                        &route,
                        (&seed, slots),
                        &mut u,
                        &seed,
                        (&mut two, &mut one),
                    )?;
                    let raised =
                        invalid_place.then(|| Fault::of_sites(LAYER_NONE, &[FaultSite::ExpertId]));
                    let replay = nodes == 1 && same(&rep, &got) && f_rep == raised;
                    let pass = empty_route
                        && same(&want, &got)
                        && f_want == raised
                        && f_got == raised
                        && replay;
                    println!(
                        "gemm32 case=g32_nvfp4_swiglu all_host slots={n_slots} \
                         invalid_place={invalid_place} empty_route={empty_route} \
                         planes_eq_two_launches={} fault_two={f_want:?} fault_fold={f_got:?} \
                         fault_graph={f_rep:?} replay_eq_eager={replay} {}",
                        same(&want, &got),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // The fold's refusals, each by its own check before any launch:
            // activations of another K than the up rows, fewer columns than
            // the slots, and a short gate or places buffer.
            let mut x = GemmAct32::new(stream, 8, k)?;
            dev.g32.enqueue_quantize_gemm32(
                stream,
                &DeviceBuffer::from_host(stream, &activations(k, 8, 9890))?,
                8,
                &mut x,
                sink,
            )?;
            let mut route = GemmRoute::new(stream, 8, n_exp)?;
            dev.gk.enqueue_route_dense(stream, 8, &mut route, sink)?;
            let mut u = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
            for (what, act_k, act_cols, g_len, sel_len, names) in [
                ("act_k", rows + 32, 8, 8 * rows, 8, "K = 160"),
                ("act_cols", rows, 4, 8 * rows, 8, "act.cols() = 4"),
                ("short_g", rows, 8, 8 * rows - 1, 8, "g.len() 1023"),
                ("short_sel", rows, 8, 8 * rows, 7, "sel.len() 7"),
            ] {
                let mut out = GemmAct32::new(stream, act_cols, act_k)?;
                let g = DeviceBuffer::<f32>::zeroed(stream, g_len)?;
                let sel = DeviceBuffer::<u32>::zeroed(stream, sel_len)?;
                let err = dev.g32.enqueue_up_swiglu32(
                    stream,
                    Gemm32Args {
                        w: Gemm32Weight::Nvfp4Planes(nv),
                        rows_per_expert: rows,
                        act: &x,
                        route: &route,
                        input: GemmInput::PerSlot,
                        y: &mut u,
                    },
                    &g,
                    SwigluSlots::Card {
                        sel: &sel,
                        n_card: n_exp,
                    },
                    &mut out,
                    sink,
                );
                let text = err
                    .as_ref()
                    .err()
                    .map_or_else(|| "none".to_string(), ToString::to_string);
                let pass = matches!(&err, Err(GpuError::Shape { .. })) && text.contains(names);
                println!(
                    "gemm32 case=g32_nvfp4_swiglu refuse_{what} error=\"{text}\" {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            stream.synchronize()?;
            let fault = gpu.take_fault()?;
            let pass = fault.is_none();
            println!(
                "gemm32 case=g32_nvfp4_swiglu refusals_launched_nothing fault={fault:?} {}",
                verdict(pass)
            );
            Ok(ok && pass)
        }

        /// The SwiGLU epilogue of the up stacks beside NVFP4 that declare one
        /// against their two launches: module doc, `g32_swiglu_epi`.
        fn swiglu_epi_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_swiglu_epi") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            let same =
                |a: &Planes32, b: &Planes32| a.0 == b.0 && nan_bits_equal(&a.1, &b.1) && a.2 == b.2;
            let k = 512usize;
            // Q4_K over a remapped card route (`gemm_q4k_swiglu`): every third
            // of 24 experts the host's, four slots a token sharing its column.
            let (n_map, top_k) = (24usize, 4usize);
            let map: Vec<u32> = (0..n_map)
                .map(|x| if x % 3 == 2 { HOST } else { (x - x / 3) as u32 })
                .collect();
            let n_card = n_map - n_map / 3;
            let map_d = DeviceBuffer::from_host(stream, &map)?;
            let max_t = NV_MS[NV_MS.len() - 1];
            let max_slots = max_t * top_k;
            let n_sb = k / 256;
            for rows in [640usize, 160] {
                let n_rows = n_card * rows;
                let words = bytes_to_words(&synthetic(
                    GemmWeight::Q4K,
                    n_rows * n_sb,
                    0x5200 + rows as u64,
                ));
                let w = DeviceTensor::upload(stream, &words, n_rows, 36 * n_sb)?;
                let mut route = GemmRoute::new(stream, max_slots, n_card)?;
                let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut sel_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut x = GemmAct::new(stream, max_t, k)?;
                let mut u = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                let seed = DeviceBuffer::from_host(stream, &activations(rows, max_slots, 9811))?;
                let mut two = GemmAct32::new(stream, max_slots, rows)?;
                let mut one = GemmAct32::new(stream, max_slots, rows)?;
                for &t in &NV_MS {
                    let n_slots = t * top_k;
                    // From 64 tokens on, token 0's column is scaled under the
                    // F32 walk's activation scale floor, so the tiles it lands
                    // in walk again exactly.
                    let tiny = t >= 64;
                    let ids =
                        route_ids(Routing::Uniform, t, top_k, n_map, 0x5b ^ (t + rows) as u64);
                    let sel: Vec<u32> = ids.iter().map(|&i| map[i as usize]).collect();
                    let mut xs = activations(k, t, 9910 + t as u32);
                    if tiny {
                        xs[..k].iter_mut().for_each(|v| *v *= 1e-25);
                    }
                    gpu.enqueue_quantize_gemm(
                        &DeviceBuffer::from_host(stream, &xs)?,
                        t,
                        &mut x,
                        sink,
                    )?;
                    let gh = activations(rows, n_slots, 9710 + t as u32);
                    let mut run = |ids: &[u32], sel: &[u32], gh: &[f32]| {
                        ids_d.copy_from_host(stream, &pad(ids, max_slots))?;
                        sel_d.copy_from_host(stream, &pad(sel, max_slots))?;
                        dev.g32.enqueue_route_remap(
                            stream, &ids_d, &map_d, n_slots, &mut route, sink,
                        )?;
                        let g = DeviceBuffer::from_host(stream, gh)?;
                        let slots = SwigluSlots::Card {
                            sel: &sel_d,
                            n_card,
                        };
                        swiglu_sides(
                            dev,
                            UpW::Grouped(GemmWeight::Q4K, &w, &x),
                            (rows, GemmInput::Shared { top_k }),
                            &route,
                            (&g, slots),
                            &mut u,
                            &seed,
                            (&mut two, &mut one),
                        )
                    };
                    let ([(want, f_want), (got, f_got), (rep, f_rep)], nodes) =
                        run(&ids, &sel, &gh)?;
                    let replay = nodes == 1 && same(&rep, &got) && f_rep == f_got;
                    let pass = same(&want, &got) && f_want.is_none() && f_got.is_none() && replay;
                    println!(
                        "gemm32 case=g32_swiglu_epi up=q4k slots=card rows={rows} K={k} T={t} \
                         n_slots={n_slots} tiny_scale_col={tiny} planes_eq_two_launches={} \
                         faults={f_want:?}/{f_got:?} replay_eq_eager={replay} act_fnv={:016x} {}",
                        same(&want, &got),
                        Fnv1a64::default().u32s(&got.0).value(),
                        verdict(pass)
                    );
                    ok &= pass;
                    if t != 64 {
                        continue;
                    }
                    // Faults, each case alone: a NaN gate value in a card
                    // column; then a slot the route refuses on a card place,
                    // one on a place no side serves and a listed slot whose
                    // place is in [n_card, HOST).
                    let card: Vec<usize> =
                        (0..n_slots).filter(|&s| sel[s] < n_card as u32).collect();
                    for nan_gate in [true, false] {
                        let (mut ids_f, mut sel_f, mut gh_f) =
                            (ids.clone(), sel.clone(), gh.clone());
                        let sites: &[FaultSite] = if nan_gate {
                            gh_f[card[0] * rows + 7] = f32::NAN;
                            &[FaultSite::QuantColumn]
                        } else {
                            ids_f[card[1]] = n_map as u32;
                            ids_f[card[2]] = n_map as u32 + 5;
                            sel_f[card[2]] = n_card as u32;
                            sel_f[card[3]] = n_card as u32 + 1;
                            &[FaultSite::ExpertId, FaultSite::QuantColumn]
                        };
                        let ([(want, f_want), (got, f_got), (rep, f_rep)], nodes) =
                            run(&ids_f, &sel_f, &gh_f)?;
                        let replay = nodes == 1 && same(&rep, &got) && f_rep == f_got;
                        let raised = Some(Fault::of_sites(LAYER_NONE, sites));
                        let pass =
                            same(&want, &got) && f_want == raised && f_got == raised && replay;
                        println!(
                            "gemm32 case=g32_swiglu_epi up=q4k rows={rows} T={t} \
                             nan_gate={nan_gate} planes_eq_two_launches={} fault_two={f_want:?} \
                             fault_fold={f_got:?} replay_eq_eager={replay} {}",
                            same(&want, &got),
                            verdict(pass)
                        );
                        ok &= pass;
                    }
                }
            }
            // Q8_0 planes over a dense one-expert table, every slot
            // (`gemm_q8_0p_swiglu_all` and, from a wave of the slab entry on,
            // `gemm_q8_0p_slab_swiglu_all`).
            let rows = 640usize;
            let (wts, _) = weights(false, rows, k, 0x5300)?;
            let res = Res::new(dev, &wts, false, rows)?;
            let (qs, d) = res
                .plane
                .as_ref()
                .ok_or("gate_gemm: a Q8_0 stack has planes")?;
            let wave = dev.g32.q8_0p_wave() as usize;
            let big = GEMM_BN * wave.div_ceil(rows.div_ceil(128));
            let mut slabs = Vec::new();
            for m in [9usize, big] {
                // The slab entry's choice for this launch (`q8_0p_slab`): the
                // grid is the route's row slabs times its most tiles.
                let grid = rows.div_ceil(128) * (m / GEMM_BN + 1);
                let slab = dev.g32.q8_0p_slab(grid as u32, m as u32, 1);
                slabs.push(slab);
                let mut route = GemmRoute::new(stream, m, 1)?;
                dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                let mut xs = activations(k, m, 9920 + m as u32);
                xs[..k].iter_mut().for_each(|v| *v *= 1e-25);
                let mut x = GemmAct32::new(stream, m, k)?;
                dev.g32.enqueue_quantize_gemm32(
                    stream,
                    &DeviceBuffer::from_host(stream, &xs)?,
                    m,
                    &mut x,
                    sink,
                )?;
                let mut u = DeviceBuffer::<f32>::zeroed(stream, m * rows)?;
                let seed = DeviceBuffer::from_host(stream, &activations(rows, m, 9821))?;
                let mut two = GemmAct32::new(stream, m, rows)?;
                let mut one = GemmAct32::new(stream, m, rows)?;
                for (nan_gate, refused) in [(false, false), (true, false), (false, true)] {
                    // The route raises a refused slot's `ExpertId`; the GEMMs
                    // over it write the slot's up row NaN, so either side
                    // raises `QuantColumn` alone.
                    let mut route_ok = true;
                    if refused {
                        let mut ids = vec![0u32; m];
                        ids[m - 1] = 1;
                        let ids = DeviceBuffer::from_host(stream, &ids)?;
                        dev.gk.enqueue_route(stream, &ids, m, &mut route, sink)?;
                        stream.synchronize()?;
                        route_ok = gpu.take_fault()?
                            == Some(Fault::of_sites(LAYER_NONE, &[FaultSite::ExpertId]));
                    }
                    let mut gh = activations(rows, m, 9720 + m as u32);
                    if nan_gate {
                        gh[(m - 1) * rows + 5] = f32::NAN;
                    }
                    let g = DeviceBuffer::from_host(stream, &gh)?;
                    let ([(want, f_want), (got, f_got), (rep, f_rep)], nodes) = swiglu_sides(
                        dev,
                        UpW::G32(Gemm32Weight::Q8_0Plane { qs, d }, &x),
                        (rows, GemmInput::PerSlot),
                        &route,
                        (&g, SwigluSlots::All),
                        &mut u,
                        &seed,
                        (&mut two, &mut one),
                    )?;
                    let raised = (nan_gate || refused)
                        .then(|| Fault::of_sites(LAYER_NONE, &[FaultSite::QuantColumn]));
                    let replay = nodes == 1 && same(&rep, &got) && f_rep == f_got;
                    let pass = route_ok
                        && same(&want, &got)
                        && f_want == raised
                        && f_got == raised
                        && replay;
                    println!(
                        "gemm32 case=g32_swiglu_epi up=q8_0p slots=all rows={rows} K={k} m={m} \
                         slab={slab} nan_gate={nan_gate} refused={refused} route_ok={route_ok} \
                         planes_eq_two_launches={} \
                         faults={f_want:?}/{f_got:?} replay_eq_eager={replay} act_fnv={:016x} {}",
                        same(&want, &got),
                        Fnv1a64::default().u32s(&got.0).value(),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            let pass = slabs == [false, true];
            println!(
                "gemm32 case=g32_swiglu_epi q8_0p_entries slab={slabs:?} wave={wave} {}",
                verdict(pass)
            );
            Ok(ok && pass)
        }

        /// Nemotron's Q5_0 sites in the GEMM: `ssm_in` (17504 × 3136) and
        /// `ffn_up` (12544 × 3136), and 272 rows (two full slabs and one of
        /// 16) at K 1056 (33 blocks: an odd count, a second 32-block window
        /// holding one).
        const Q50_SHAPES: [(usize, usize); 3] = [(17504, 3136), (12544, 3136), (ROWS, 1056)];
        /// Columns per `q5_0p_is_the_gemv` run: the prompt's U = 512 and the
        /// edges of a 64-column tile and of the gemv's 8-column chunk.
        const Q50_US: [usize; 5] = [1, 8, 9, 64, 512];
        /// The share of the derived band (`gemm32::q5_0p_gemv_roundings`) a
        /// GEMM output may sit from the gemv's: tighter than the bound, which
        /// counts every rounding at its worst.
        // PIN(2026-10-16): worst 0.116 of the band (272 × K 1056, U 512), the Nemotron shapes ≤ 0.066.
        const Q50_PIN: f64 = 0.25;

        /// `n_rows` rows of Q5_0 blocks from `seed` (finite f16 `d`, random
        /// `qh` and nibbles), decoded to the signed codes `q − 16` and `d`;
        /// the decode is checked against ggml's `dequant_row` bit for bit
        /// (`d·(q − 16)` is one f32 product on both sides).
        fn weights_q50(n_rows: usize, k: usize, seed: u64) -> Result<(Wts, bool), GateError> {
            let mut g = Lcg(seed);
            let kb = k / 32;
            let mut bytes = vec![0u8; n_rows * kb * 22];
            let (mut d, mut q) = (
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * k),
            );
            for blk in bytes.as_chunks_mut::<22>().0 {
                for b in blk.iter_mut() {
                    *b = g.next() as u8;
                }
                let neg = g.next() & 1 == 1;
                let db = f16_bits(&mut g, neg);
                blk[0..2].copy_from_slice(&db.to_le_bytes());
                d.push(half_to_f32(db));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let mut v = [0i8; 32];
                for (j, &b) in blk[6..22].iter().enumerate() {
                    v[j] = ((b & 0x0f) | ((((qh >> j) & 1) as u8) << 4)) as i8 - 16;
                    v[16 + j] = ((b >> 4) | ((((qh >> (16 + j)) & 1) as u8) << 4)) as i8 - 16;
                }
                q.extend_from_slice(&v);
            }
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(
                    GgmlType::Q5_0,
                    &bytes[r * kb * 22..(r + 1) * kb * 22],
                    &mut want,
                )?;
                for (v, &w) in want.iter().enumerate() {
                    let mine = d[r * kb + v / 32] * f32::from(q[r * k + v]);
                    decode_ok &= mine.to_bits() == w.to_bits();
                }
            }
            Ok((
                Wts {
                    k,
                    bytes,
                    d,
                    m: vec![0.0; n_rows * kb],
                    q,
                },
                decode_ok,
            ))
        }

        /// One `q5_0p_is_the_gemv` run's comparison: outputs checked, off
        /// the contract's bits, outside the gemv band; values that are not
        /// finite on either side; the worst distance over the band, the
        /// outputs bit-equal to the gemv's, and the first failure.
        struct GemvCmp {
            checked: usize,
            bits_off: usize,
            band_off: usize,
            nonfinite: usize,
            worst: f64,
            same_bits: usize,
            first: Option<String>,
        }

        /// The GEMM's `y` (token-major, `y[s · rows + r]`) against the
        /// gemv's (`yv`, the same layout) over `u` columns, each checked
        /// output against [`dot32`]'s contract bits and the gemv band.
        fn gemv_cmp(
            w: &Wts,
            rows: usize,
            a: &HostAct,
            y: &[f32],
            yv: &[f32],
            u: usize,
        ) -> Result<GemvCmp, GateError> {
            let kb = w.k / 32;
            let rel = Q50_PIN * gamma(bloomery_gpu_gates::gemm32::q5_0p_gemv_roundings(kb));
            let parts = par_slots(
                u,
                || GemvCmp {
                    checked: 0,
                    bits_off: 0,
                    band_off: 0,
                    nonfinite: 0,
                    worst: 0.0,
                    same_bits: 0,
                    first: None,
                },
                |s, c: &mut GemvCmp| {
                    for r in 0..rows {
                        let (got, gv) = (y[s * rows + r], yv[s * rows + r]);
                        if !got.is_finite() || !gv.is_finite() {
                            c.nonfinite += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: gemm {got:e} gemv {gv:e}")
                            });
                            continue;
                        }
                        if !row_checked(u, s, r) {
                            continue;
                        }
                        let (want, _, mag) = row_dot(w, r, a, s, false);
                        let band = rel * mag;
                        let err = (f64::from(got) - f64::from(gv)).abs();
                        c.checked += 1;
                        c.same_bits += usize::from(got.to_bits() == gv.to_bits());
                        c.worst = c.worst.max(if band > 0.0 { err / band } else { err });
                        if err > band {
                            c.band_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!(
                                    "slot {s} row {r}: gemm {got:e} gemv {gv:e} band {band:.3e}"
                                )
                            });
                        }
                        if got.to_bits() != want.to_bits() {
                            c.bits_off += 1;
                            c.first.get_or_insert_with(|| {
                                format!("slot {s} row {r}: gemm {got:e} contract {want:e}")
                            });
                        }
                    }
                },
            )?;
            let mut all = GemvCmp {
                checked: 0,
                bits_off: 0,
                band_off: 0,
                nonfinite: 0,
                worst: 0.0,
                same_bits: 0,
                first: None,
            };
            for p in parts {
                all.checked += p.checked;
                all.bits_off += p.bits_off;
                all.band_off += p.band_off;
                all.nonfinite += p.nonfinite;
                all.worst = all.worst.max(p.worst);
                all.same_bits += p.same_bits;
                all.first = all.first.or(p.first);
            }
            Ok(all)
        }

        /// `q5_0p_is_the_gemv`: the resident packed Q5_0 (`pack_q5_0`, the
        /// tensor `enqueue_gemv_q5_0` reads) in `gemm_q5_0p` over the dense
        /// table, at each of [`Q50_SHAPES`] and [`Q50_US`] columns, against
        /// the Q5_0 gemv run eight columns at a time on the same input: both
        /// quantizers' codes, scales and sums the host's; every output of
        /// both written and finite; every checked output bit for bit the
        /// contract's transcription and within the gemv band
        /// (`gemm32::q5_0p_gemv_roundings`, scaled by [`Q50_PIN`]), and its
        /// BF16 store the F32 cells rounded to nearest even. Its refusal at
        /// another K is `g32_refuse`'s.
        fn q5_0p_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("q5_0p_is_the_gemv") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let max = Q50_US[Q50_US.len() - 1];
            let mut route = GemmRoute::new(stream, max, 1)?;
            let mut ok = true;
            for (si, (rows, k)) in Q50_SHAPES.into_iter().enumerate() {
                let kb = k / 32;
                let (w, dec) = weights_q50(rows, k, 0x3500 + si as u64)?;
                let packed = pack_q5_0(&w.bytes, k, rows)?;
                let words = packed.len() / rows;
                let wt = DeviceTensor::upload(stream, &packed, rows, words)?;
                let mut act = GemmAct32::new(stream, max, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
                let mut yg = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
                for u in Q50_US {
                    let x = activations(k, u, 9800 + (k * 3 + u + si) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, u, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, u, &mut route, sink)?;
                    // The gemv's scratch quantizes all its columns: one of `u`.
                    let mut act8u = Q8Blocks32::with_slots(stream, k, u)?;
                    gpu.q5()
                        .enqueue_quantize_q8(stream, &xd, &mut act8u, sink)?;
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    dev.g32.enqueue_gemm32(
                        stream,
                        Gemm32Args {
                            w: Gemm32Weight::Q5_0Packed(&wt),
                            rows_per_expert: rows,
                            act: &act,
                            route: &route,
                            input: GemmInput::PerSlot,
                            y: &mut y,
                        },
                    )?;
                    stream.synchronize()?;
                    let mut got = y.to_host_vec(stream)?;
                    got.truncate(u * rows);
                    // The gemv, eight columns a launch, laid token-major.
                    let mut yv = vec![SENT; u * rows];
                    for c0 in (0..u).step_by(8) {
                        let m = (u - c0).min(8);
                        yg.copy_from_host(stream, &vec![SENT; yg.len()])?;
                        gpu.q5()
                            .enqueue_gemv_q5_0(stream, &wt, &act8u, 0, rows, c0, m, &mut yg, 0)?;
                        let g = yg.to_host_vec(stream)?;
                        for c in 0..m {
                            for r in 0..rows {
                                yv[(c0 + c) * rows + r] = g[r * m + c];
                            }
                        }
                    }
                    let (ha, want) = host_act(&x, k, u);
                    let act_ok = planes_equal(&act, u, &want, stream)?;
                    let h8 = act8u.readback(stream)?;
                    let qst = 256 * kb.div_ceil(32);
                    let act8_ok = (0..u).all(|c| {
                        (0..kb).all(|b| {
                            (0..8).all(|i| {
                                h8.q[c * qst + 256 * (b >> 5) + 32 * i + (b & 31)]
                                    == want.0[c * 16 * act.steps() + 8 * b + i]
                            }) && h8.d8[c * kb + b].to_bits() == ha.d[c * kb + b].to_bits()
                                && h8.s8[c * kb + b] == ha.s[c * kb + b]
                        })
                    });
                    let unwritten = got
                        .iter()
                        .chain(&yv)
                        .filter(|v| v.to_bits() == SENT.to_bits())
                        .count();
                    let GemvCmp {
                        checked,
                        bits_off,
                        band_off,
                        nonfinite,
                        worst,
                        same_bits,
                        first,
                    } = gemv_cmp(&w, rows, &ha, &got, &yv, u)?;
                    let wgt = Gemm32Weight::Q5_0Packed(&wt);
                    let bf16 = bf16_product32(dev, wgt, rows, (&act, &route), &got)?;
                    let fault = gpu.take_fault()?;
                    let pass = dec
                        && act_ok
                        && act8_ok
                        && bf16
                        && unwritten == 0
                        && nonfinite == 0
                        && checked > 0
                        && bits_off == 0
                        && band_off == 0
                        && fault.is_none();
                    println!(
                        "gemm32 case=q5_0p_is_the_gemv rows={rows} K={k} U={u} decode_vs_ggml={dec} \
                         act_eq_host={act_ok} q8blocks32_eq_host={act8_ok} unwritten={unwritten} \
                         nonfinite={nonfinite} checked={checked} contract_bits_differ={bits_off} \
                         gemv_band_off={band_off} worst_over_gemv_band={worst:.4} \
                         gemv_bits_equal={same_bits} bf16_eq_f32_rne={bf16} y_fnv={:016x} {}",
                        Fnv1a64::default().f32s(&got).value(),
                        verdict(pass)
                    );
                    if let Some(f) = first {
                        println!(
                            "gemm32 case=q5_0p_is_the_gemv rows={rows} K={k} U={u} first_failure: {f}"
                        );
                    }
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// The IQ4_NL down of the Qwen3.8 card route (`gemm_iq4nl`): a
        /// synthetic stack of 64 experts × 2560 rows at K 640 (random nibble
        /// codes, finite f16 `d`), each slot its own column, the four routings
        /// at TOKENS; the decode against ggml's `dequant_row`, every checked
        /// output in its band and bit for bit the contract with `y_fnv`, a rerun
        /// bit-identical, a NaN `d` block's outputs NaN, and the host API's
        /// refusal of the stack at another K; each run's BF16 store the F32
        /// cells rounded to nearest even.
        fn iq4nl_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g38_iq4nl") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (n_exp, rows, k) = (64usize, 2560usize, 640usize);
            let (w, dec) = weights_iq4nl(n_exp * rows, k, 0x3600)?;
            let res = Res::new(dev, &w, true, n_exp * rows)?;
            let max_slots = TOKENS[TOKENS.len() - 1];
            let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            let mut act = GemmAct32::new(stream, max_slots, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
            let mut ok = true;
            gpu.clear_fault()?;
            for &t in &TOKENS {
                for (ri, &r) in ROUTINGS.iter().enumerate() {
                    let ids = route_ids(r, t, 1, n_exp, 0x39 ^ (t as u64) << 6 ^ ri as u64);
                    let n_slots = ids.len();
                    let x = activations(k, n_slots, 9900 + (t + ri) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;
                    ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                    dev.gk
                        .enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                    let got = gemm(
                        dev,
                        &res,
                        Lay::Iq4Nl,
                        rows,
                        &act,
                        &route,
                        GemmInput::PerSlot,
                        &mut y,
                        n_slots,
                    )?;
                    let (ha, _) = host_act(&x, k, n_slots);
                    let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                    let Checked {
                        checked,
                        band_off,
                        bits_off,
                        worst,
                        first,
                        ..
                    } = check(
                        &w,
                        false,
                        rows,
                        &ha,
                        &|s| Some(ids[s] as usize),
                        &|s| s,
                        &got,
                        n_slots,
                    )?;
                    let wgt = res.weight(Lay::Iq4Nl)?;
                    let bf16 = bf16_product32(dev, wgt, rows, (&act, &route), &got)?;
                    let fault = gpu.take_fault()?;
                    let pass = dec
                        && unwritten == 0
                        && band_off == 0
                        && bits_off == 0
                        && bf16
                        && fault.is_none();
                    println!(
                        "gemm32 case=g38_iq4nl down T={t} routing={r:?} slots={n_slots} \
                         decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                         worst_err_over_band={worst:.3} band_off={band_off} \
                         contract_bits_differ={bits_off} bf16_eq_f32_rne={bf16} y_fnv={:016x} {}",
                        Fnv1a64::default().f32s(&got).value(),
                        verdict(pass)
                    );
                    if let Some(f) = first {
                        println!("gemm32 case=g38_iq4nl down T={t} first_failure: {f}");
                    }
                    ok &= pass;
                }
            }
            // A rerun at 64 tokens is bit for bit the first run.
            {
                let t = 64;
                let ids = route_ids(Routing::Uniform, t, 1, n_exp, 0x3a);
                let x = activations(k, ids.len(), 9950);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, ids.len(), &mut act, sink)?;
                let a = gemm(
                    dev,
                    &res,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let b = gemm(
                    dev,
                    &res,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let same = bits_equal(&a, &b);
                println!("gemm32 case=g38_iq4nl rerun_bits={same} {}", verdict(same));
                ok &= same;
            }
            // A NaN f16 `d` in block 0 of expert 0's row 0: the outputs of
            // the slots on that expert's row 0 are NaN, every other output
            // finite — no silent failure.
            {
                let mut bytes = w.bytes.clone();
                bytes[0..2].copy_from_slice(&0x7e00u16.to_le_bytes());
                let wn = Wts {
                    k,
                    bytes,
                    d: w.d.clone(),
                    m: w.m.clone(),
                    q: w.q,
                };
                let resn = Res::new(dev, &wn, true, n_exp * rows)?;
                let ids = [0u32; 8];
                let x = activations(k, ids.len(), 9960);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, ids.len(), &mut act, sink)?;
                let got = gemm(
                    dev,
                    &resn,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let row0_nan = (0..ids.len()).all(|s| got[s * rows].is_nan());
                let rest_finite = (0..ids.len()).all(|s| {
                    got[s * rows + 1..(s + 1) * rows]
                        .iter()
                        .all(|v| v.is_finite())
                });
                let pass = row0_nan && rest_finite;
                println!(
                    "gemm32 case=g38_iq4nl fault=nan_d row0_nan={row0_nan} other_rows_finite={rest_finite} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // The host API refuses the stack at another K's words.
            {
                let mut other = GemmAct32::new(stream, 8, 672)?;
                let xd = DeviceBuffer::from_host(stream, &activations(672, 8, 9970))?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, 8, &mut other, sink)?;
                let wgt = res.weight(Lay::Iq4Nl)?;
                let mut y8 = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
                let err = dev.g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w: wgt,
                        rows_per_expert: rows,
                        act: &other,
                        route: &route,
                        input: GemmInput::PerSlot,
                        y: &mut y8,
                    },
                );
                let pass = err.is_err_and(|e| e.to_string().contains("IQ4_NL rows at K = 672"));
                println!(
                    "gemm32 case=g38_iq4nl refusal_other_k_refused={pass} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The remapped route: a map of 48 experts, every third on the card
        /// (slot `15 − e/3`, so the map reverses them) and the rest [`HOST`],
        /// under a Q5_1 stack of the 16 card experts. Per token count: the
        /// table's listed slots and tiles the host's grouping of the mapped
        /// ids with the host slots taken out, nothing refused, every host
        /// slot's rows still [`SENT`] and every card slot's rows the
        /// contract's bits. An identity map gives `gemm_route`'s table bit
        /// for bit. Then the faults: an id past the map and a map value past
        /// the stack each raise `ExpertId`, their slots listed as refused and
        /// NaN, every other output as the clean run's.
        fn remap_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_remap") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (n_map, n_card, rows, k, top_k) = (48usize, 16usize, ROWS, 640usize, 4usize);
            let map: Vec<u32> = (0..n_map)
                .map(|e| {
                    if e % 3 == 0 {
                        (n_card - 1 - e / 3) as u32
                    } else {
                        HOST
                    }
                })
                .collect();
            let (w, dec) = weights(true, n_card * rows, k, 0x3400)?;
            let res = Res::new(dev, &w, true, n_card * rows)?;
            let tokens = [1usize, 17, 512, 2048];
            let max_slots = tokens[tokens.len() - 1] * top_k;
            let mut route = GemmRoute::new(stream, max_slots, n_card)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            let map_d = DeviceBuffer::from_host(stream, &map)?;
            let mut act = GemmAct32::new(stream, max_slots, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
            let mut ok = true;
            gpu.clear_fault()?;
            let mut last = None;
            for t in tokens {
                let ids = route_ids(Routing::Uniform, t, top_k, n_map, 0x88 ^ t as u64);
                let n_slots = ids.len();
                let x = activations(k, n_slots, 9800 + t as u32);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_d, n_slots, &mut route, sink)?;
                let got = gemm(
                    dev,
                    &res,
                    Lay::Q51,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    n_slots,
                )?;
                let mapped: Vec<u32> = ids.iter().map(|&e| map[e as usize]).collect();
                let (want_cols, want_tiles) = route_ref(&mapped, n_card);
                let (cols, tiles) = route.read_back(stream)?;
                let refused = route.refused_back(stream)?;
                let table_ok = tiles == want_tiles
                    && cols[..want_cols.len()] == want_cols[..]
                    && refused.is_empty();
                let (ha, _) = host_act(&x, k, n_slots);
                let host_slots = mapped.iter().filter(|&&v| v == HOST).count();
                let Checked {
                    checked,
                    band_off,
                    bits_off,
                    host_bad,
                    worst,
                    first,
                } = check(
                    &w,
                    true,
                    rows,
                    &ha,
                    &|s| (mapped[s] != HOST).then_some(mapped[s] as usize),
                    &|s| s,
                    &got,
                    n_slots,
                )?;
                let fault = gpu.take_fault()?;
                let pass = dec
                    && table_ok
                    && host_bad == 0
                    && band_off == 0
                    && bits_off == 0
                    && fault.is_none();
                println!(
                    "gemm32 case=g32_remap T={t} slots={n_slots} host_slots={host_slots} listed={} tiles={} \
                     table_eq_host={table_ok} host_untouched={} checked={checked} worst_err_over_band={worst:.3} \
                     band_off={band_off} contract_bits_differ={bits_off} {}",
                    want_cols.len(),
                    tiles.len(),
                    host_bad == 0,
                    verdict(pass)
                );
                if let Some(f) = first {
                    println!("gemm32 case=g32_remap T={t} first_failure: {f}");
                }
                ok &= pass;
                last = Some((ids, got));
            }
            let (ids, clean) = last.ok_or("no remap run")?;

            // The K-quant family over the same remapped table: a Q4_K stack of
            // the 16 card experts (the routed gate·up's type). Host slots keep
            // the sentinel; every card slot is bit for bit the same GEMM over
            // `gemm_route`'s table of the mapped ids, each host id replaced by
            // card slot 0 — an output's bits do not depend on its tile.
            {
                let (kq, rq) = (2048usize, 256usize);
                let n_slots = ids.len();
                let bytes = synthetic(GemmWeight::Q4K, n_card * rq * kq / 256, 0x3410);
                let words = bytes_to_words(&bytes);
                let wq =
                    DeviceTensor::upload(stream, &words, n_card * rq, words.len() / (n_card * rq))?;
                let mut aq = GemmAct::new(stream, n_slots, kq)?;
                let xq = DeviceBuffer::from_host(stream, &activations(kq, n_slots, 9870))?;
                gpu.enqueue_quantize_gemm(&xq, n_slots, &mut aq, sink)?;
                let mut yq = DeviceBuffer::<f32>::zeroed(stream, n_slots * rq)?;
                let mut run_q4k = |route: &GemmRoute| -> Result<Vec<f32>, GateError> {
                    yq.copy_from_host(stream, &vec![SENT; yq.len()])?;
                    dev.gk.enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: GemmWeight::Q4K,
                            w: &wq,
                            rows_per_expert: rq,
                            act: &aq,
                            route,
                            input: GemmInput::PerSlot,
                            y: &mut yq,
                        },
                    )?;
                    stream.synchronize()?;
                    Ok(yq.to_host_vec(stream)?)
                };
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_d, n_slots, &mut route, sink)?;
                let y_remap = run_q4k(&route)?;
                let mapped: Vec<u32> = ids.iter().map(|&e| map[e as usize]).collect();
                let plain_ids: Vec<u32> = mapped
                    .iter()
                    .map(|&v| if v == HOST { 0 } else { v })
                    .collect();
                let mut plain = GemmRoute::new(stream, max_slots, n_card)?;
                ids_d.copy_from_host(stream, &pad(&plain_ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, n_slots, &mut plain, sink)?;
                let y_plain = run_q4k(&plain)?;
                let (mut host_ok, mut card_ok) = (true, true);
                for s in 0..n_slots {
                    let (a, b) = (
                        &y_remap[s * rq..(s + 1) * rq],
                        &y_plain[s * rq..(s + 1) * rq],
                    );
                    if mapped[s] == HOST {
                        host_ok &= a.iter().all(|v| v.to_bits() == SENT.to_bits());
                    } else {
                        card_ok &=
                            bits_equal(a, b) && a.iter().all(|v| v.to_bits() != SENT.to_bits());
                    }
                }
                let fault = gpu.take_fault()?;
                let pass = host_ok && card_ok && fault.is_none();
                println!(
                    "gemm32 case=g32_remap gemm_q4k slots={n_slots} host_untouched={host_ok} \
                     card_eq_gemm_route_table={card_ok} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            let n_slots = ids.len();
            let x = activations(k, n_slots, 9800 + 2048);
            let xd = DeviceBuffer::from_host(stream, &x)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;

            // An identity map is `gemm_route`'s table.
            {
                let ident: Vec<u32> = (0..n_card as u32).collect();
                let ident_d = DeviceBuffer::from_host(stream, &ident)?;
                let ids16 = route_ids(Routing::Uniform, 1000, top_k, n_card, 0x99);
                ids_d.copy_from_host(stream, &pad(&ids16, ids_d.len()))?;
                dev.g32.enqueue_route_remap(
                    stream,
                    &ids_d,
                    &ident_d,
                    ids16.len(),
                    &mut route,
                    sink,
                )?;
                let a = route.read_back(stream)?;
                let mut plain = GemmRoute::new(stream, max_slots, n_card)?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids16.len(), &mut plain, sink)?;
                let b = plain.read_back(stream)?;
                let pass = a == b && gpu.take_fault()?.is_none();
                println!(
                    "gemm32 case=g32_remap identity_map slots={} table_eq_gemm_route={} {}",
                    ids16.len(),
                    a == b,
                    verdict(pass)
                );
                ok &= pass;
            }

            // Faults: slot 5's id past the map; then every slot of card expert
            // 3 (map value 16, past the stack).
            let want = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
            for case in ["id_past_map", "value_past_stack"] {
                let (ids_f, map_f) = if case == "id_past_map" {
                    let mut v = ids.clone();
                    v[5] = n_map as u32 + 2;
                    (v, map.clone())
                } else {
                    let mut mp = map.clone();
                    mp[3] = n_card as u32;
                    (ids.clone(), mp)
                };
                let bad: Vec<usize> = (0..n_slots)
                    .filter(|&s| {
                        ids_f[s] as usize >= n_map || map_f[ids_f[s] as usize] == n_card as u32
                    })
                    .collect();
                let map_fd = DeviceBuffer::from_host(stream, &map_f)?;
                ids_d.copy_from_host(stream, &pad(&ids_f, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_fd, n_slots, &mut route, sink)?;
                let got = gemm(
                    dev,
                    &res,
                    Lay::Q51,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    n_slots,
                )?;
                let fault = gpu.take_fault()?;
                let refused = route.refused_back(stream)?;
                let listed_ok = refused.iter().map(|&s| s as usize).collect::<Vec<_>>() == bad;
                let nan_ok = bad
                    .iter()
                    .all(|&s| got[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
                let others = (0..n_slots).filter(|s| !bad.contains(s)).all(|s| {
                    bits_equal(
                        &got[s * rows..(s + 1) * rows],
                        &clean[s * rows..(s + 1) * rows],
                    )
                });
                let pass = fault == want && listed_ok && nan_ok && others && !bad.is_empty();
                println!(
                    "gemm32 case=g32_remap fault={case} bad_slots={} fault=\"{}\" refused_listed={listed_ok} \
                     refused_nan={nan_ok} others_eq_clean={others} {}",
                    bad.len(),
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The wide F32 product against `f32_gemv` launches of up to eight
        /// columns, bit for bit, at the Qwen3.8 row counts (96, 128, 512) and
        /// one that is not a multiple of 32 (100); a column count one short of
        /// a ubatch (4095) ragged at both edges.
        fn f32tile_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_f32tile") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let k = 2560usize;
            let mut ok = true;
            for (rows, ns) in [
                (96usize, &[1usize, 8, 9, 512, 4095][..]),
                (100, &[1, 9, 512][..]),
                (128, &[1, 8, 9, 512, 4095][..]),
                (512, &[1, 9, 512][..]),
            ] {
                let wh = activations(k, rows, 9900 + rows as u32);
                let w = DeviceTensor::upload(stream, &wh, rows, k)?;
                for &n in ns {
                    let xh = activations(k, n, 9950 + n as u32);
                    let xd = DeviceBuffer::from_host(stream, &xh)?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 64])?;
                    dev.g32.enqueue_f32_tile(stream, &w, &xd, n, &mut y)?;
                    stream.synchronize()?;
                    let got = y.to_host_vec(stream)?;
                    let tail_ok = got[rows * n..]
                        .iter()
                        .all(|v| v.to_bits() == SENT.to_bits());
                    let mut differ = 0usize;
                    let mut yg = DeviceBuffer::<f32>::zeroed(stream, rows * 8)?;
                    for c0 in (0..n).step_by(8) {
                        let m = (n - c0).min(8);
                        let xc = DeviceBuffer::from_host(stream, &xh[c0 * k..(c0 + m) * k])?;
                        gpu.q8f32().enqueue_f32_gemv(stream, &w, &xc, m, &mut yg)?;
                        let g = yg.to_host_vec(stream)?;
                        for c in 0..m {
                            for r in 0..rows {
                                if got[(c0 + c) * rows + r].to_bits() != g[r * m + c].to_bits() {
                                    differ += 1;
                                }
                            }
                        }
                    }
                    let pass = differ == 0 && tail_ok;
                    println!(
                        "gemm32 case=g32_f32tile rows={rows} K={k} n={n} bits_differ_vs_f32_gemv={differ} \
                         past_end_untouched={tail_ok} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// The host API's refusals, each a named error: a case passes only
        /// when its call is an error whose text holds the case's own fragment
        /// (another check refusing first, or a launch error, does not).
        fn refusals(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_refuse") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (k, rows) = (320usize, 32usize);
            let (w8, _) = weights(false, rows, k, 0x3500)?;
            let r8 = Res::new(dev, &w8, false, rows)?;
            let x = DeviceBuffer::from_host(stream, &activations(k, 8, 2))?;
            let short = DeviceBuffer::from_host(stream, &activations(k, 1, 3))?;
            // `act` quantized for all 8 columns; `act4` for 4 of its 8;
            // `narrow` holds 4; `act_q` takes the quantizer's refusals.
            let mut act = GemmAct32::new(stream, 8, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 8, &mut act, sink)?;
            let mut act4 = GemmAct32::new(stream, 8, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 4, &mut act4, sink)?;
            let mut narrow = GemmAct32::new(stream, 4, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 4, &mut narrow, sink)?;
            let mut act_q = GemmAct32::new(stream, 8, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
            let mut y_short = DeviceBuffer::<f32>::zeroed(stream, 8 * rows - 1)?;
            let unfilled = GemmRoute::new(stream, 8, 1)?;
            let mut route = GemmRoute::new(stream, 8, 1)?;
            dev.gk.enqueue_route_dense(stream, 8, &mut route, sink)?;
            let mut remap_route = GemmRoute::new(stream, 8, 1)?;
            let w96 = DeviceTensor::upload(stream, &activations(96, 4, 1), 4, 96)?;
            let w320 = DeviceTensor::upload(stream, &activations(k, 4, 4), 4, k)?;
            let empty = DeviceBuffer::<u32>::zeroed(stream, 0)?;
            let map = DeviceBuffer::from_host(stream, &[0u32])?;
            let ids = DeviceBuffer::<u32>::zeroed(stream, 9)?;
            let plane = r8.weight(Lay::Q8Plane)?;
            let (qs, _) = r8.plane.as_ref().ok_or("a Q8_0 stack with no planes")?;
            // Packed Q5_0 rows at K 96: 17 words a row where K 320 needs 55.
            let (w50, _) = weights_q50(rows, 96, 0x35f0)?;
            let p50 = pack_q5_0(&w50.bytes, 96, rows)?;
            let q50_96 = DeviceTensor::upload(stream, &p50, rows, p50.len() / rows)?;
            // NVFP4 planes at K 384, and an act at K 96 (a multiple of 32 but
            // not of 64): K 96 against them refuses on K, K 384 with a short
            // `qs`, a short `e` or two scales for the one-expert table on the
            // planes themselves.
            let (wnv, _) = weights_nvfp4(rows, 384, 0x36f0)?;
            let pnv = repack::<Nvfp4>(&wnv.bytes, rows, 384)?;
            let nvq = DeviceTensor::upload(stream, &pnv.qs, rows, 384 / 8)?;
            let nve = DeviceTensor::upload(stream, &pnv.e, rows, 384 / 64)?;
            let nvs = DeviceTensor::upload(stream, &[0.5f32], 1, 1)?;
            let nvs2 = DeviceTensor::upload(stream, &[0.5f32; 2], 2, 1)?;
            // The short planes upload their leading columns only, so the
            // refusal the clause tests is the gemm's, not `upload`'s length
            // check.
            let nvq_short =
                DeviceTensor::upload(stream, &pnv.qs[..rows * (384 / 8 - 1)], rows, 384 / 8 - 1)?;
            let nve_short =
                DeviceTensor::upload(stream, &pnv.e[..rows * (384 / 64 - 1)], rows, 384 / 64 - 1)?;
            let mut act96 = GemmAct32::new(stream, 8, 96)?;
            dev.g32.enqueue_quantize_gemm32(
                stream,
                &DeviceBuffer::from_host(stream, &activations(96, 8, 6))?,
                8,
                &mut act96,
                sink,
            )?;
            let mut act384 = GemmAct32::new(stream, 8, 384)?;
            dev.g32.enqueue_quantize_gemm32(
                stream,
                &DeviceBuffer::from_host(stream, &activations(384, 8, 5))?,
                8,
                &mut act384,
                sink,
            )?;
            // The scales of a stack one block wider than the codes.
            let d_wide =
                DeviceTensor::upload(stream, &vec![0u16; rows * (k / 32 + 1)], rows, k / 32 + 1)?;
            // `x` from its second value: 4 bytes past a 16-byte boundary.
            let x_off = bloomery_gpu::Window::<f32>::of(&x, 4, 4 * k)?;
            let gemm = |w: Gemm32Weight<'_>,
                        rows_per_expert: usize,
                        act: &GemmAct32,
                        route: &GemmRoute,
                        y: &mut DeviceBuffer<f32>| {
                dev.g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w,
                        rows_per_expert,
                        act,
                        route,
                        input: GemmInput::PerSlot,
                        y,
                    },
                )
            };
            let unit = |r: Result<GemmAct32, bloomery_gpu::GpuError>| r.map(|_| ());
            let cases: Vec<(&str, Result<(), bloomery_gpu::GpuError>, &str)> = vec![
                (
                    "act_k_48",
                    unit(GemmAct32::new(stream, 8, 48)),
                    "k must be a multiple of 32",
                ),
                (
                    "act_cols_0",
                    unit(GemmAct32::new(stream, 0, k)),
                    "1 <= cols <=",
                ),
                (
                    "quant_short_input",
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &short, 8, &mut act_q, sink),
                    "< n_cols*k",
                ),
                (
                    "quant_cols_past_act",
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &x, 9, &mut act_q, sink),
                    "1 <= n_cols <= act.cols()",
                ),
                (
                    "swiglu_act_limit_nan",
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &x,
                        &x,
                        Act::SwigluClamp { limit: f32::NAN },
                        8,
                        &mut act_q,
                        sink,
                    ),
                    "the clamp limit must be finite",
                ),
                (
                    "swiglu_act_short_input",
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &short,
                        &x,
                        Act::SiluMul,
                        8,
                        &mut act_q,
                        sink,
                    ),
                    "need n_cols*k",
                ),
                (
                    "unfilled_route",
                    gemm(plane, rows, &act, &unfilled, &mut y),
                    "a filled route table",
                ),
                (
                    "q5_0p_other_k",
                    gemm(
                        Gemm32Weight::Q5_0Packed(&q50_96),
                        rows,
                        &act,
                        &route,
                        &mut y,
                    ),
                    "packed Q5_0 rows at K = 320 are 55 words",
                ),
                (
                    "rows_not_16",
                    gemm(plane, 24, &act, &route, &mut y),
                    "rows_per_expert must be a positive multiple of 16",
                ),
                (
                    "q5_1_layout_at_q8_0_words",
                    gemm(Gemm32Weight::Q5_1File(&r8.file), rows, &act, &route, &mut y),
                    "Q5_1 rows at K =",
                ),
                (
                    "iq4nl_layout_at_q8_0_words",
                    gemm(
                        Gemm32Weight::Iq4NlFile(&r8.file),
                        rows,
                        &act,
                        &route,
                        &mut y,
                    ),
                    "IQ4_NL rows at K =",
                ),
                (
                    "plane_shape",
                    gemm(
                        Gemm32Weight::Q8_0Plane { qs, d: &d_wide },
                        rows,
                        &act,
                        &route,
                        &mut y,
                    ),
                    "Q8_0 planes at K =",
                ),
                (
                    "act_cols_short",
                    gemm(plane, rows, &narrow, &route, &mut y),
                    "activation columns, act holds 4",
                ),
                (
                    "act_quantized_short",
                    gemm(plane, rows, &act4, &route, &mut y),
                    "the last quantizer launch wrote 4",
                ),
                (
                    "y_short",
                    gemm(plane, rows, &act, &route, &mut y_short),
                    "< n_slots*rows_per_expert",
                ),
                (
                    "remap_empty_map",
                    dev.g32
                        .enqueue_route_remap(stream, &ids, &empty, 8, &mut remap_route, sink),
                    "the map holds 1 to",
                ),
                (
                    "remap_slots_past_table",
                    dev.g32
                        .enqueue_route_remap(stream, &ids, &map, 9, &mut remap_route, sink),
                    "1 <= n_slots <= the table's 8 slots",
                ),
                (
                    "f32_tile_k_96",
                    dev.g32.enqueue_f32_tile(stream, &w96, &x, 1, &mut y),
                    "k must be a positive multiple of 64",
                ),
                (
                    "f32_tile_short_x",
                    dev.g32.enqueue_f32_tile(stream, &w320, &short, 2, &mut y),
                    "x.len() ",
                ),
                (
                    "f32_tile_x_unaligned",
                    dev.g32.enqueue_f32_tile(stream, &w320, &x_off, 1, &mut y),
                    "must be 16-byte aligned",
                ),
                (
                    "nvfp4_k_not_64",
                    gemm(
                        Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                            qs: &nvq,
                            e: &nve,
                            scales: &nvs,
                        }),
                        rows,
                        &act96,
                        &route,
                        &mut y,
                    ),
                    "NVFP4 stack at K=96",
                ),
                (
                    "nvfp4_qs_short",
                    gemm(
                        Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                            qs: &nvq_short,
                            e: &nve,
                            scales: &nvs,
                        }),
                        rows,
                        &act384,
                        &route,
                        &mut y,
                    ),
                    "NVFP4 stack at K=384",
                ),
                (
                    "nvfp4_e_short",
                    gemm(
                        Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                            qs: &nvq,
                            e: &nve_short,
                            scales: &nvs,
                        }),
                        rows,
                        &act384,
                        &route,
                        &mut y,
                    ),
                    "NVFP4 stack at K=384",
                ),
                (
                    // PIN(2026-10-20): retire the GEMM NaN-scale clause; gate_nvfp4 covers tensor_scales' finite-scale refusal.
                    "nvfp4_scales_count",
                    gemm(
                        Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                            qs: &nvq,
                            e: &nve,
                            scales: &nvs2,
                        }),
                        rows,
                        &act384,
                        &route,
                        &mut y,
                    ),
                    "and 2 scales",
                ),
            ];
            let all = cases
                .iter()
                .all(|(_, got, want)| matches!(got, Err(e) if e.to_string().contains(want)));
            println!(
                "gemm32 case=g32_refuse {} {}",
                cases
                    .iter()
                    .map(|(n, got, want)| match got {
                        Err(e) if e.to_string().contains(want) => format!("{n}=refused"),
                        Err(e) => format!("{n}=REFUSED-OTHERWISE({e})"),
                        Ok(()) => format!("{n}=ACCEPTED"),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                verdict(all)
            );
            Ok(all)
        }

        /// The product epilogue equals the separate projection and norm,
        /// including exact replay, faults, graph replay and token-unit cuts,
        /// for a BF16 weight and, on a native block-scaled build, an MXFP8
        /// weight's planes as stored, whose epilogue also writes the
        /// output's MXFP8 form: a next MXFP8 product reads it bit for bit as
        /// it reads `enqueue_mxfp8_cols`' form of the split output.
        pub(super) fn norm_gate_case(gpu: &Gpu, timed: bool) -> Result<bool, GateError> {
            if !timed && !wanted("zgate") {
                return Ok(true);
            }
            use crate::bench_arm::{N, burst, print_arm, replay};
            use bloomery_gpu::gemm::{MmaForm, Mxfp8NormGateArgs};
            use bloomery_gpu::linear::norm_gate::{
                GdnGate, NormGateArgs, NormGateKernels, NormGateTail,
            };
            use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight};
            use gguf::quant::fp32_to_bf16_bits;
            let stream = gpu.stream();
            let g32 = Gemm32Kernels::load(gpu.context())?;
            let norm = NormGateKernels::load(gpu.context())?;
            let sink = gpu.unlabelled_sink();
            let bundles = bloomery_gpu_gates::ptx::current_exe_bundles()?;
            let native = bloomery_gpu_gates::ptx::modules(&bundles)?.iter().any(|m| {
                bloomery_gpu_gates::ptx::body(std::slice::from_ref(m), "mxfp8_w8a8_norm_gate")
                    .is_some()
            });
            let forms: &[MmaForm] = if native {
                &[MmaForm::Bf16, MmaForm::Mxfp8]
            } else {
                &[MmaForm::Bf16]
            };
            let mut ok = true;
            for &form in forms {
                let mx = form == MmaForm::Mxfp8;
                let cases: &[(usize, usize, usize, &str)] = if timed {
                    &[(6144, 2560, 4095, "normal")]
                } else {
                    &[
                        (128, 64, 1, "normal"),
                        (256, 128, 8, "normal"),
                        (256, 128, 9, "normal"),
                        (256, 192, 129, "normal"),
                        (6144, 2560, 4095, "normal"),
                        (256, 128, 9, "nan_x"),
                        (256, 128, 9, "range_x"),
                        (256, 128, 9, "overflow_z"),
                        (256, 128, 9, "nan_o"),
                        (256, 128, 9, "square_overflow"),
                        (256, 128, 9, "nan_gain"),
                        (256, 128, 9, "big_gain"),
                    ]
                };
                for &(rows, k, n, edge) in cases {
                    // An MXFP8 column holds whole 128-value scale words.
                    let k = if mx { k.next_multiple_of(128) } else { k };
                    // MXFP8: random E4M3 codes (no NaN code) on block scales
                    // 2^-7 .. 2^0; overflow_z puts row 0's first block past the
                    // tensor cores' range, so the row replays to an infinite z.
                    let mx_planes = |rows: usize,
                                     k: usize,
                                     edit: &dyn Fn(&mut [u8], &mut [u8])|
                     -> Result<_, GateError> {
                        let mut rng = bloomery_gpu_gates::GateRng(0x7a67_6174 + (rows * k) as u64);
                        let mut codes: Vec<u8> = (0..rows * k)
                            .map(|_| (rng.below(0x7f) as u8) | ((rng.below(2) as u8) << 7))
                            .collect();
                        let mut scales: Vec<u8> = (0..rows * k / 32)
                            .map(|_| 120 + rng.below(8) as u8)
                            .collect();
                        edit(&mut codes, &mut scales);
                        Ok((
                            DeviceTensor::upload(stream, &bytes_to_words(&codes), rows, k / 4)?,
                            DeviceTensor::upload(stream, &bytes_to_words(&scales), rows, k / 128)?,
                        ))
                    };
                    let (w, e) = if mx {
                        let (qs, e) = mx_planes(rows, k, &|codes, scales| {
                            if edge == "overflow_z" {
                                codes[..k].fill(0);
                                codes[0] = 0x7e;
                                codes[1] = 0x7e;
                                scales[0] = 254;
                            }
                        })?;
                        (qs, Some(e))
                    } else {
                        let mut bits: Vec<u16> = activations(k, rows, 1101)
                            .into_iter()
                            .map(fp32_to_bf16_bits)
                            .collect();
                        if edge == "overflow_z" {
                            bits[..k].fill(0);
                            bits[0] = 0x7f00;
                            bits[1] = 0x7f00;
                        }
                        let words: Vec<u32> = bits
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
                            .collect();
                        (DeviceTensor::upload(stream, &words, rows, k / 2)?, None)
                    };
                    let planes = e.as_ref().map(|e| E4m3Weight {
                        qs: &w,
                        e,
                        scales: E4m3Scales::Mxfp8,
                    });
                    let mut xh = activations(k, n, 1102);
                    let mut oh = activations(rows, n, 1103);
                    let mut gh = activations(128, 1, 1104);
                    match edge {
                        "nan_x" => xh[8 * k + 31] = f32::NAN,
                        "range_x" => xh[8 * k + 31] = f32::from_bits(0x200),
                        "overflow_z" => {
                            xh[8 * k] = 1.0;
                            xh[8 * k + 1] = 1.0;
                        }
                        "nan_o" => oh[8 * rows + 127] = f32::NAN,
                        "square_overflow" => oh[8 * rows + 127] = f32::MAX,
                        "nan_gain" => gh[127] = f32::NAN,
                        // Row 0 of every head leaves the tensor cores' range
                        // except column 8's first head, so its output form
                        // refuses each column by its other heads.
                        "big_gain" => {
                            gh[0] = 2f32.powi(60);
                            oh[8 * rows] = 0.0;
                        }
                        _ => {}
                    }
                    let x = DeviceBuffer::from_host(stream, &xh)?;
                    let o = DeviceBuffer::from_host(stream, &oh)?;
                    let gain = DeviceBuffer::from_host(stream, &gh)?;
                    let formed = |xx: &DeviceBuffer<f32>,
                                  count: usize|
                     -> Result<GemmAct32, GateError> {
                        let mut aa = GemmAct32::new(stream, count, k)?.with_form(stream, form)?;
                        g32.enqueue_form_cols(stream, form, xx, count, &mut aa)?;
                        Ok(aa)
                    };
                    let act = formed(&x, n)?;
                    let product =
                        |s: &cuda_core::CudaStream, z: &mut DeviceBuffer<f32>| match planes {
                            Some(p) => g32.enqueue_mxfp8_w8a8(s, p, &x, &act, n, z),
                            None => g32.enqueue_bf16_mma_cols(s, &w, &x, &act, n, z),
                        };
                    // The output's MXFP8 form, which only the MXFP8 entry writes.
                    let room = |count: usize| -> Result<GemmAct32, GateError> {
                        Ok(GemmAct32::new(stream, count, rows)?
                            .with_form(stream, MmaForm::Mxfp8)?)
                    };
                    let fused = |s: &cuda_core::CudaStream,
                                 (xx, aa, count): (&DeviceBuffer<f32>, &GemmAct32, usize),
                                 o: &DeviceBuffer<f32>,
                                 z: &mut DeviceBuffer<f32>,
                                 y: &mut DeviceBuffer<f32>,
                                 out: &mut GemmAct32| {
                        let tail = NormGateTail {
                            o,
                            w: &gain,
                            gate: GdnGate::Sigmoid,
                            eps: 1e-6,
                            fault: sink,
                            y,
                        };
                        match planes {
                            Some(w) => g32.enqueue_mxfp8_w8a8_norm_gate(
                                s,
                                Mxfp8NormGateArgs {
                                    w,
                                    x: xx,
                                    act: aa,
                                    n_cols: count,
                                    z,
                                    tail,
                                    out,
                                },
                            ),
                            None => g32.enqueue_bf16_mma_norm_gate(s, &w, xx, aa, count, z, tail),
                        }
                    };
                    let mut z = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 32])?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 32])?;
                    let mut split = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 32])?;
                    gpu.clear_fault()?;
                    product(stream, &mut z)?;
                    norm.enqueue_norm_gate(
                        stream,
                        NormGateArgs {
                            o: &o,
                            z: &z,
                            w: &gain,
                            gate: GdnGate::Sigmoid,
                            eps: 1e-6,
                            n_v: rows / 128,
                            m: n,
                            fault: sink,
                            y: &mut split,
                        },
                    )?;
                    stream.synchronize()?;
                    let expected_fault = gpu.take_fault()?;
                    let expected = split.to_host_vec(stream)?;
                    let overflow_reached =
                        edge != "overflow_z" || z.to_host_vec(stream)?[8 * rows].is_infinite();
                    z.copy_from_host(stream, &vec![SENT; z.len()])?;
                    let mut out = room(n)?;
                    let enqueue = |s: &cuda_core::CudaStream,
                                   z: &mut DeviceBuffer<f32>,
                                   y: &mut DeviceBuffer<f32>,
                                   out: &mut GemmAct32| {
                        fused(s, (&x, &act, n), &o, z, y, out)
                    };
                    gpu.clear_fault()?;
                    enqueue(stream, &mut z, &mut y, &mut out)?;
                    stream.synchronize()?;
                    let fault = gpu.take_fault()?;
                    let eager = y.to_host_vec(stream)?;
                    let same = nan_bits_equal(&eager, &expected);
                    // The next MXFP8 product over an epilogue's form against
                    // the one over the split output's quantizer form.
                    let next = if mx {
                        let (qs2, e2) = mx_planes(128, rows, &|_, _| {})?;
                        let mut quantized = room(n)?;
                        g32.enqueue_mxfp8_cols(stream, &split, n, &mut quantized)?;
                        let mut want = DeviceBuffer::from_host(stream, &vec![SENT; 128 * n])?;
                        let w = E4m3Weight {
                            qs: &qs2,
                            e: &e2,
                            scales: E4m3Scales::Mxfp8,
                        };
                        g32.enqueue_mxfp8_w8a8(stream, w, &split, &quantized, n, &mut want)?;
                        let want = want.to_host_vec(stream)?;
                        Some((qs2, e2, want))
                    } else {
                        None
                    };
                    let next_bits =
                        |y: &DeviceBuffer<f32>, out: &GemmAct32| -> Result<bool, GateError> {
                            let Some((qs, e, want)) = &next else {
                                return Ok(true);
                            };
                            let w = E4m3Weight {
                                qs,
                                e,
                                scales: E4m3Scales::Mxfp8,
                            };
                            let mut got = DeviceBuffer::from_host(stream, &vec![SENT; 128 * n])?;
                            Ok(g32
                                .enqueue_mxfp8_w8a8(stream, w, y, out, n, &mut got)
                                .is_ok()
                                && nan_bits_equal(&got.to_host_vec(stream)?, want))
                        };
                    let form_bits = next_bits(&y, &out)?;
                    let scratch_kept = edge == "nan_x"
                        || edge == "range_x"
                        || edge == "overflow_z"
                        || z.to_host_vec(stream)?
                            .iter()
                            .all(|v| v.to_bits() == SENT.to_bits());
                    let graph = gpu.capture(|s| enqueue(s, &mut z, &mut y, &mut out))?;
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    graph.launch(stream)?;
                    stream.synchronize()?;
                    let graph_bits = bits_equal(&eager, &y.to_host_vec(stream)?);
                    let mut cut_bits = true;
                    if n == 9 && edge == "normal" {
                        let mut cut = Vec::new();
                        for (start, count) in [(0, 1), (1, 8)] {
                            let xx = DeviceBuffer::from_host(
                                stream,
                                &xh[start * k..(start + count) * k],
                            )?;
                            let oo = DeviceBuffer::from_host(
                                stream,
                                &oh[start * rows..(start + count) * rows],
                            )?;
                            let aa = formed(&xx, count)?;
                            let mut zz = DeviceBuffer::<f32>::zeroed(stream, rows * count)?;
                            let mut yy = DeviceBuffer::<f32>::zeroed(stream, rows * count)?;
                            let mut out = room(count)?;
                            fused(stream, (&xx, &aa, count), &oo, &mut zz, &mut yy, &mut out)?;
                            cut.extend(yy.to_host_vec(stream)?);
                        }
                        cut_bits = bits_equal(&cut, &eager[..rows * n]);
                    }
                    // One output scratch through a launch whose subnormal
                    // outputs refuse its columns and then this case's launch,
                    // eagerly and replayed: the scratch holds the second
                    // launch's form alone.
                    let mut reuse_bits = true;
                    if mx && n == 9 && edge == "normal" {
                        let tiny = DeviceBuffer::from_host(stream, &vec![1e-42f32; rows * n])?;
                        let mut reused = room(n)?;
                        let both = |s: &cuda_core::CudaStream,
                                    z: &mut DeviceBuffer<f32>,
                                    y: &mut DeviceBuffer<f32>,
                                    out: &mut GemmAct32| {
                            fused(s, (&x, &act, n), &tiny, z, y, out)?;
                            fused(s, (&x, &act, n), &o, z, y, out)
                        };
                        both(stream, &mut z, &mut y, &mut reused)?;
                        stream.synchronize()?;
                        reuse_bits = next_bits(&y, &reused)?;
                        let graph = gpu.capture(|s| both(s, &mut z, &mut y, &mut reused))?;
                        graph.launch(stream)?;
                        stream.synchronize()?;
                        reuse_bits &= next_bits(&y, &reused)?;
                    }
                    let fault_expected = matches!(
                        edge,
                        "nan_x" | "overflow_z" | "nan_o" | "square_overflow" | "nan_gain"
                    );
                    let fault_ok = fault == expected_fault
                        && fault
                            == fault_expected.then(|| Fault::at(LAYER_NONE, FaultSite::LinearGate));
                    let pass = same
                        && form_bits
                        && graph_bits
                        && scratch_kept
                        && cut_bits
                        && reuse_bits
                        && fault_ok
                        && overflow_reached;
                    println!(
                        "zgate form={form:?} rows={rows} K={k} M={n} edge={edge} split_bits={same} out_form_bits={form_bits} eager_graph={graph_bits} scratch_kept={scratch_kept} unit_cut={cut_bits} scratch_reuse={reuse_bits} fault_ok={fault_ok} overflow_reached={overflow_reached} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                    if timed {
                        for joined in [false, true] {
                            // The split MXFP8 arm also quantizes the output
                            // for the next site, which the epilogue writes.
                            let mut enq = |s: &cuda_core::CudaStream| {
                                if joined {
                                    enqueue(s, &mut z, &mut y, &mut out)
                                } else {
                                    product(s, &mut z)?;
                                    norm.enqueue_norm_gate(
                                        s,
                                        NormGateArgs {
                                            o: &o,
                                            z: &z,
                                            w: &gain,
                                            gate: GdnGate::Sigmoid,
                                            eps: 1e-6,
                                            n_v: rows / 128,
                                            m: n,
                                            fault: sink,
                                            y: &mut y,
                                        },
                                    )?;
                                    if mx {
                                        g32.enqueue_mxfp8_cols(s, &y, n, &mut out)?;
                                    }
                                    Ok(())
                                }
                            };
                            let eager = burst(stream, &mut enq)?;
                            let graph = gpu.capture(|s| {
                                for _ in 0..N {
                                    enq(s)?;
                                }
                                Ok(())
                            })?;
                            let graph = replay(stream, &graph)?;
                            let arm = match (mx, joined) {
                                (false, false) => "zgate_split",
                                (false, true) => "zgate_fused",
                                (true, false) => "zgate_mxfp8_split",
                                (true, true) => "zgate_mxfp8_fused",
                            };
                            // Graph nodes: the MXFP8 epilogue's refusal
                            // memset precedes its kernel.
                            let nodes = match (mx, joined) {
                                (false, true) => 1,
                                (true, true) | (false, false) => 2,
                                (true, false) => 3,
                            };
                            print_arm(arm, n, 0, nodes, eager, graph);
                        }
                    }
                }
            }
            let x = DeviceBuffer::from_host(stream, &vec![1.0f32; 128])?;
            let o = DeviceBuffer::from_host(stream, &vec![1.0f32; 128])?;
            let gain = DeviceBuffer::from_host(stream, &vec![1.0f32; 128])?;
            let mut z = DeviceBuffer::<f32>::zeroed(stream, 128)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 128)?;
            let says = |r: Result<(), GpuError>, need: &str| {
                r.is_err_and(|e| e.to_string().contains(need))
            };
            let partial = |r: Result<(), GpuError>| says(r, "whole 128-value heads");
            // The column and room checks `GemmAct32::room` owns, as each
            // quantizer reaches them: no columns, more than the scratch holds
            // and a scratch without the form are refused; all of its columns
            // pass.
            let cols_refused = |form: MmaForm| -> Result<bool, GateError> {
                let quantize =
                    |act: &mut GemmAct32, n: usize| g32.enqueue_form_cols(stream, form, &x, n, act);
                let mut act = GemmAct32::new(stream, 1, 128)?.with_form(stream, form)?;
                let none = says(quantize(&mut act, 0), "got 0");
                let past = says(quantize(&mut act, 2), "act.cols() = 1, got 2");
                let no_room = says(
                    quantize(&mut GemmAct32::new(stream, 1, 128)?, 1),
                    "with_form",
                );
                let all = quantize(&mut act, 1).is_ok();
                let pass = none && past && no_room && all;
                println!(
                    "zgate case=cols_refused form={form:?} no_cols={none} past_cols={past} no_room={no_room} all_cols={all} {}",
                    verdict(pass)
                );
                Ok(pass)
            };
            let mut cols_ok = cols_refused(MmaForm::Bf16)?;
            let mut act = GemmAct32::new(stream, 1, 64)?.with_form(stream, MmaForm::Bf16)?;
            g32.enqueue_bf16_cols(stream, &x, 1, &mut act)?;
            let w = DeviceTensor::upload(stream, &vec![0x3f80_3f80u32; 127 * 32], 127, 32)?;
            let mut partial_head = partial(g32.enqueue_bf16_mma_norm_gate(
                stream,
                &w,
                &x,
                &act,
                1,
                &mut z,
                NormGateTail {
                    o: &o,
                    w: &gain,
                    gate: GdnGate::Sigmoid,
                    eps: 1e-6,
                    fault: sink,
                    y: &mut y,
                },
            ));
            // The epilogue's norm gates by sigmoid alone: a whole head under
            // a SiLU tail is refused by name, and the site runs that norm
            // apart.
            let head = DeviceTensor::upload(stream, &vec![0x3f80_3f80u32; 128 * 32], 128, 32)?;
            let silu = says(
                g32.enqueue_bf16_mma_norm_gate(
                    stream,
                    &head,
                    &x,
                    &act,
                    1,
                    &mut z,
                    NormGateTail {
                        o: &o,
                        w: &gain,
                        gate: GdnGate::Silu,
                        eps: 1e-6,
                        fault: sink,
                        y: &mut y,
                    },
                ),
                "gates by sigmoid",
            );
            // The MXFP8 entry refuses as the product entry does: a partial
            // head, a misaligned value plane, other scales and a stale form;
            // and an output scratch of another K or without the MXFP8 form.
            let mut mx_refused = true;
            if native {
                let mut act = GemmAct32::new(stream, 1, 128)?.with_form(stream, MmaForm::Mxfp8)?;
                g32.enqueue_mxfp8_cols(stream, &x, 1, &mut act)?;
                let mut bf16 = GemmAct32::new(stream, 1, 128)?.with_form(stream, MmaForm::Bf16)?;
                g32.enqueue_bf16_cols(stream, &x, 1, &mut bf16)?;
                let upload = |rows: usize| -> Result<_, GateError> {
                    Ok((
                        DeviceTensor::upload(stream, &vec![0x3838_3838u32; rows * 32], rows, 32)?,
                        DeviceTensor::upload(stream, &vec![0x7f7f_7f7fu32; rows], rows, 1)?,
                    ))
                };
                let (qs127, e127) = upload(127)?;
                let (qs, e) = upload(128)?;
                let words = DeviceBuffer::from_host(stream, &vec![0x3838_3838u32; 128 * 32 + 1])?;
                let shifted = DeviceTensor::<u32>::window_of(&words, 4, 128, 32)?;
                let mut out = GemmAct32::new(stream, 1, 128)?.with_form(stream, MmaForm::Mxfp8)?;
                let mut out_k =
                    GemmAct32::new(stream, 1, 256)?.with_form(stream, MmaForm::Mxfp8)?;
                let mut out_bf16 =
                    GemmAct32::new(stream, 1, 128)?.with_form(stream, MmaForm::Bf16)?;
                let mut run = |qs: &DeviceTensor<u32>,
                               e: &DeviceTensor<u32>,
                               scales: E4m3Scales,
                               act: &GemmAct32,
                               out: &mut GemmAct32| {
                    g32.enqueue_mxfp8_w8a8_norm_gate(
                        stream,
                        Mxfp8NormGateArgs {
                            w: E4m3Weight { qs, e, scales },
                            x: &x,
                            act,
                            n_cols: 1,
                            z: &mut z,
                            tail: NormGateTail {
                                o: &o,
                                w: &gain,
                                gate: GdnGate::Sigmoid,
                                eps: 1e-6,
                                fault: sink,
                                y: &mut y,
                            },
                            out,
                        },
                    )
                };
                let mx = E4m3Scales::Mxfp8;
                partial_head &= partial(run(&qs127, &e127, mx, &act, &mut out));
                let unaligned = says(
                    run(&shifted, &e, mx, &act, &mut out),
                    "is not 16-byte aligned",
                );
                let other_scales = says(
                    run(&qs, &e, E4m3Scales::Fp8B128, &act, &mut out),
                    "not MXFP8's",
                );
                let stale = says(run(&qs, &e, mx, &bf16, &mut out), "MXFP8 form");
                let other_k = says(run(&qs, &e, mx, &act, &mut out_k), "not 128");
                let no_room = says(run(&qs, &e, mx, &act, &mut out_bf16), "with_form");
                let fresh = run(&qs, &e, mx, &act, &mut out).is_ok();
                // An output scratch of fewer columns than the product's, with
                // the input, z, o and y room for all of them; the scratch of
                // all of them passes.
                let x2 = DeviceBuffer::from_host(stream, &vec![1.0f32; 256])?;
                let o2 = DeviceBuffer::from_host(stream, &vec![1.0f32; 256])?;
                let mut z2 = DeviceBuffer::<f32>::zeroed(stream, 256)?;
                let mut y2 = DeviceBuffer::<f32>::zeroed(stream, 256)?;
                let mut act2 = GemmAct32::new(stream, 2, 128)?.with_form(stream, MmaForm::Mxfp8)?;
                g32.enqueue_mxfp8_cols(stream, &x2, 2, &mut act2)?;
                let mut out2 = GemmAct32::new(stream, 2, 128)?.with_form(stream, MmaForm::Mxfp8)?;
                let mut run2 = |out: &mut GemmAct32| {
                    g32.enqueue_mxfp8_w8a8_norm_gate(
                        stream,
                        Mxfp8NormGateArgs {
                            w: E4m3Weight {
                                qs: &qs,
                                e: &e,
                                scales: mx,
                            },
                            x: &x2,
                            act: &act2,
                            n_cols: 2,
                            z: &mut z2,
                            tail: NormGateTail {
                                o: &o2,
                                w: &gain,
                                gate: GdnGate::Sigmoid,
                                eps: 1e-6,
                                fault: sink,
                                y: &mut y2,
                            },
                            out,
                        },
                    )
                };
                let out_cols = says(run2(&mut out), "act.cols() = 1, got 2");
                let out_all = run2(&mut out2).is_ok();
                mx_refused = unaligned
                    && other_scales
                    && stale
                    && other_k
                    && no_room
                    && fresh
                    && out_cols
                    && out_all;
                println!(
                    "zgate case=mxfp8_refused unaligned_weight={unaligned} other_scales={other_scales} stale_form={stale} out_other_k={other_k} out_no_room={no_room} fresh_read={fresh} out_past_cols={out_cols} out_all_cols={out_all} {}",
                    verdict(mx_refused)
                );
                cols_ok &= cols_refused(MmaForm::Mxfp8)?;
            }
            let overflow = matches!(norm.enqueue_norm_gate(stream, NormGateArgs {
                o: &o, z: &z, w: &gain, gate: GdnGate::Sigmoid, eps: 1e-6, n_v: 2, m: usize::MAX,
                fault: sink, y: &mut y,
            }), Err(e) if e.to_string().contains("overflows"));
            println!(
                "zgate native={native} partial_head_refused={partial_head} silu_refused={silu} size_overflow_refused={overflow} {}",
                verdict(partial_head && silu && overflow)
            );
            Ok(ok && partial_head && silu && overflow && mx_refused && cols_ok)
        }

        /// Ragged dense MMA tiles, tails, repeatability and graph replay.
        fn dense_mma_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("dense_mma") {
                return Ok(true);
            }
            use gguf::quant::{bf16_bits_to_f32, fp32_to_bf16_bits};
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let mut ok = true;
            // Sparse operands exercise range changes at every reduction boundary.
            type Edge<'a> = (&'a str, &'a [(usize, u32)], &'a [(usize, u32)], u32);
            const OVERFLOW: Edge<'static> = (
                "rounded_product",
                &[(0, 0x7f00_0000)],
                &[(0, 0x3fff_8000)],
                0x7f7f_8000,
            );
            let cancel = |at| {
                let mut w = CANCEL.1.to_vec();
                let mut x = CANCEL.2.to_vec();
                w[2].0 = at;
                x[2].0 = at;
                (w, x)
            };
            const CANCEL: Edge<'static> = (
                "within_fragment",
                &[(0, 0x7f00_0000), (1, 0x7f00_0000), (2, 0xff00_0000)],
                &[
                    (0, 0x3f80_0000),
                    (1, 0x3f80_0000),
                    (2, 0x3f80_0000),
                    (15, 0x0040_0000),
                ],
                0x7f00_0000,
            );
            let fragment = cancel(16);
            let stage = cancel(32);
            let pipeline = cancel(96);
            const BUTTERFLY: Edge<'static> = (
                "butterfly",
                &[
                    (0, 0x7f00_0000),
                    (1, 0xff00_0000),
                    (32, 0x7f00_0000),
                    (33, 0xff00_0000),
                ],
                &[
                    (0, 0x3f80_0000),
                    (1, 0x3f80_0000),
                    (32, 0x3f80_0000),
                    (33, 0x3f80_0000),
                ],
                0x7fc0_0000,
            );
            const STORE: Edge<'static> = (
                "store_overflow",
                &[(0, 0x7f00_0000), (1, 0x7f00_0000)],
                &[(0, 0x3f80_0000), (1, 0x3f80_0000)],
                0x7f80_0000,
            );
            let sparse = |v: &mut Vec<f32>, offset, set: &[(usize, u32)]| {
                v.fill(0.0);
                for &(i, b) in set {
                    v[offset + i] = f32::from_bits(b);
                }
            };
            for (rows, k, n, r, c, edge) in [
                (1, 64, 1, 0, 0, None),
                (17, 128, 17, 0, 0, None),
                (33, 2560, 33, 0, 0, None),
                (100, 256, 65, 0, 0, None),
                (33, 640, 513, 0, 0, None),
                (33, 64, 1100, 0, 0, None),
                (1, 128, 1, 0, 0, Some(OVERFLOW)),
                (1, 128, 1, 0, 0, Some(CANCEL)),
                (
                    1,
                    64,
                    1,
                    0,
                    0,
                    Some(("between_fragments", &fragment.0, &fragment.1, 0x7f00_0000)),
                ),
                (
                    1,
                    128,
                    1,
                    0,
                    0,
                    Some(("between_fragments", &fragment.0, &fragment.1, 0x7f00_0000)),
                ),
                (
                    1,
                    128,
                    1,
                    0,
                    0,
                    Some(("between_stages", &stage.0, &stage.1, 0x7f00_0000)),
                ),
                (
                    1,
                    128,
                    1,
                    0,
                    0,
                    Some(("pipeline_wrap", &pipeline.0, &pipeline.1, 0x7f00_0000)),
                ),
                (
                    65,
                    128,
                    65,
                    63,
                    63,
                    Some(("warp_corner", &pipeline.0, &pipeline.1, 0x7f00_0000)),
                ),
                (
                    65,
                    128,
                    65,
                    64,
                    64,
                    Some(("tile_corner", &pipeline.0, &pipeline.1, 0x7f00_0000)),
                ),
                (1, 128, 1, 0, 0, Some(BUTTERFLY)),
                (1, 128, 1, 0, 0, Some(STORE)),
            ] {
                let mut wh = activations(k, rows, 9271 + rows as u32);
                if let Some((_, w0, _, _)) = edge {
                    sparse(&mut wh, r * k, w0);
                }
                let bits: Vec<u16> = wh.into_iter().map(fp32_to_bf16_bits).collect();
                let wf: Vec<Vec<f32>> = bits
                    .chunks_exact(k)
                    .map(|r| r.iter().copied().map(bf16_bits_to_f32).collect())
                    .collect();
                let words: Vec<u32> = bits
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
                    .collect();
                let w = DeviceTensor::upload(stream, &words, rows, k / 2)?;
                let mut xh = activations(k, n, 9281 + n as u32);
                if let Some((_, _, x0, _)) = edge {
                    sparse(&mut xh, c * k, x0);
                }
                let x = DeviceBuffer::from_host(stream, &xh)?;
                let formats = if edge.is_some() && k % 128 == 0 { 3 } else { 1 };
                use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight};
                let byte_planes = if formats > 1 {
                    // Sparse weights are zero or +/-2^127: E4M3 +/-1 with scale 2^127.
                    let qs: Vec<u32> = wf
                        .iter()
                        .flat_map(|row| row.as_chunks::<4>().0.iter())
                        .map(|v| {
                            v.iter().enumerate().fold(0, |word, (i, &value)| {
                                let code = if value == 0.0 {
                                    0
                                } else if value < 0.0 {
                                    0xb8
                                } else {
                                    0x38
                                };
                                word | (code << (8 * i))
                            })
                        })
                        .collect();
                    Some((
                        DeviceTensor::upload(stream, &qs, rows, k / 4)?,
                        DeviceTensor::upload(
                            stream,
                            &vec![0xfefe_fefe; rows * k / 128],
                            rows,
                            k / 128,
                        )?,
                        DeviceTensor::upload(
                            stream,
                            &vec![0x7f00_0000; rows * k / 128],
                            rows,
                            k / 128,
                        )?,
                    ))
                } else {
                    None
                };
                for format in 0..formats {
                    let format_name = ["bf16", "mxfp8", "fp8_b128"][format];
                    let enqueue = |s: &cuda_core::CudaStream, exact, y: &mut DeviceBuffer<f32>| {
                        if format == 0 {
                            if exact {
                                gpu.q8f32().enqueue_bf16_tile(s, &w, &x, n, y)
                            } else {
                                gpu.q8f32().enqueue_bf16_mma(s, &w, &x, n, y)
                            }
                        } else {
                            let (qs, mx, fp) = byte_planes.as_ref().unwrap();
                            let weight = E4m3Weight {
                                qs,
                                e: if format == 1 { mx } else { fp },
                                scales: if format == 1 {
                                    E4m3Scales::Mxfp8
                                } else {
                                    E4m3Scales::Fp8B128
                                },
                            };
                            if exact {
                                dev.g32.enqueue_e4m3_tile(s, weight, &x, n, y)
                            } else {
                                dev.g32.enqueue_e4m3_mma(s, weight, &x, n, y)
                            }
                        }
                    };
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 64])?;
                    enqueue(stream, false, &mut y)?;
                    stream.synchronize()?;
                    let eager = y.to_host_vec(stream)?;
                    let want = edge.map(|(_, _, _, bits)| bits);
                    let name = edge.map_or("ordinary", |(name, _, _, _)| name);
                    let tag = format!(
                        "{format_name} rows={rows} K={k} n={n} probe={r},{c} edge={name} edge_bits={want:x?}"
                    );
                    let range = want.map_or_else(
                        || eager[..rows * n].iter().all(|v| v.is_finite()),
                        |bits| {
                            let expected = f32::from_bits(bits);
                            let got = eager[c * rows + r];
                            if expected.is_nan() {
                                got.is_nan()
                            } else {
                                got.to_bits() == bits
                            }
                        },
                    );
                    let quality = if edge.is_some() {
                        let mut exact = DeviceBuffer::from_host(stream, &vec![SENT; y.len()])?;
                        enqueue(stream, true, &mut exact)?;
                        let exact = exact.to_host_vec(stream)?;
                        // PIN(2026-10-23): healthy cells retain MMA values; only exceptional rows and
                        // columns match the exact tile. The cells fixture below pins that distinction.
                        let same = eager[..rows * n]
                            .iter()
                            .zip(&exact)
                            .enumerate()
                            .filter(|(i, _)| i % rows == r || i / rows == c)
                            .all(|(_, (a, b))| {
                                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
                            });
                        println!("dense_mma {tag} exact_tile_bits={same}");
                        let finite_quality = !f32::from_bits(want.unwrap()).is_finite()
                            || bloomery_gpu_gates::rounding::check_mma(
                                &tag,
                                (&eager[..rows * n], &xh, k),
                                &wf,
                            );
                        same && finite_quality
                    } else {
                        bloomery_gpu_gates::rounding::check_mma(
                            &tag,
                            (&eager[..rows * n], &xh, k),
                            &wf,
                        )
                    };
                    let tail = eager[rows * n..]
                        .iter()
                        .all(|v| v.to_bits() == SENT.to_bits());
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    let graph = gpu.capture(|s| enqueue(s, false, &mut y))?;
                    y.copy_from_host(stream, &vec![SENT; y.len()])?;
                    graph.launch(stream)?;
                    stream.synchronize()?;
                    let replay = y.to_host_vec(stream)?;
                    let same = eager
                        .iter()
                        .zip(&replay)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    println!(
                        "dense_mma {tag} range={range} past_end_untouched={tail} eager_graph_bits={same} {}",
                        verdict(range && tail && same && quality)
                    );
                    ok &= quality && range && tail && same;
                    if format == 0 {
                        // The columns' BF16 form gives the same bits, eager and
                        // replayed over a form rewritten from zeros after the
                        // capture, so the replay's own form launch counts.
                        let cols = |s: &cuda_core::CudaStream,
                                    act: &mut GemmAct32,
                                    y: &mut DeviceBuffer<f32>| {
                            dev.g32.enqueue_bf16_cols(s, &x, n, act)?;
                            dev.g32.enqueue_bf16_mma_cols(s, &w, &x, act, n, y)
                        };
                        let mut act = GemmAct32::new(stream, n, k)?
                            .with_form(stream, bloomery_gpu::gemm::MmaForm::Bf16)?;
                        y.copy_from_host(stream, &vec![SENT; y.len()])?;
                        cols(stream, &mut act, &mut y)?;
                        let got = y.to_host_vec(stream)?;
                        y.copy_from_host(stream, &vec![SENT; y.len()])?;
                        let graph = gpu.capture(|s| cols(s, &mut act, &mut y))?;
                        let zeros = DeviceBuffer::from_host(stream, &vec![0.0f32; n * k])?;
                        dev.g32.enqueue_bf16_cols(stream, &zeros, n, &mut act)?;
                        graph.launch(stream)?;
                        stream.synchronize()?;
                        let replay = y.to_host_vec(stream)?;
                        let bits = |v: &[f32]| {
                            v.iter()
                                .zip(&eager)
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        };
                        let (eager_same, graph_same) = (bits(&got), bits(&replay));
                        println!(
                            "dense_mma bf16_cols rows={rows} K={k} n={n} edge={name} \
                             eager_bits={eager_same} graph_bits={graph_same} {}",
                            verdict(eager_same && graph_same)
                        );
                        ok &= eager_same && graph_same;
                    }
                }
            }
            // Each cell's path, visible with unit weights against activations
            // below BF16's last bit (1 + 3·2^-10, which BF16 rounds to 1): K on
            // the tensor cores and K·(1 + 3·2^-10) on the exact owner. A
            // subnormal activation in one column, then also weights of 2^60 in
            // one row, send that column's and that row's cells to the exact
            // tile's bits and leave every other cell of their tiles at K. The
            // subnormal column's 127·(1 + 3·2^-10) lies nearer BF16's step
            // above, so a BF16 store that truncates misses its bits.
            {
                use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight};
                let (rows, k, n, odd_row, odd_col) = (130, 128, 130, 64, 129);
                let x_of = |odd: bool| {
                    let mut x = vec![f32::from_bits(0x3f80_6000); n * k];
                    if odd {
                        x[odd_col * k] = f32::from_bits(0x200);
                    }
                    DeviceBuffer::from_host(stream, &x)
                };
                // A row's words, its MXFP8 and its FP8_B128 scale words: 1 or 2^60.
                let planes = |odd: bool| {
                    let row = |r| odd && r == odd_row;
                    let of = |normal: u32, big: u32, cols: usize| {
                        let v: Vec<u32> = (0..rows)
                            .flat_map(|r| {
                                std::iter::repeat_n(if row(r) { big } else { normal }, cols)
                            })
                            .collect();
                        DeviceTensor::upload(stream, &v, rows, cols)
                    };
                    Ok::<_, GpuError>((
                        of(0x3f80_3f80, 0x5d80_5d80, k / 2)?,
                        DeviceTensor::upload(
                            stream,
                            &vec![0x3838_3838u32; rows * k / 4],
                            rows,
                            k / 4,
                        )?,
                        of(0x7f7f_7f7f, 0xbbbb_bbbb, k / 128)?,
                        of(0x3f80_0000, 0x5d80_0000, k / 128)?,
                    ))
                };
                for (odd_r, odd_c) in [(false, false), (false, true), (true, true)] {
                    let x = x_of(odd_c)?;
                    let (w, qs, mx, fp) = planes(odd_r)?;
                    for (format, scales) in [
                        ("bf16", None),
                        ("mxfp8", Some((&mx, E4m3Scales::Mxfp8))),
                        ("fp8_b128", Some((&fp, E4m3Scales::Fp8B128))),
                    ] {
                        let run = |exact: bool| {
                            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n])?;
                            match scales {
                                None if exact => {
                                    gpu.q8f32().enqueue_bf16_tile(stream, &w, &x, n, &mut y)?
                                }
                                None => gpu.q8f32().enqueue_bf16_mma(stream, &w, &x, n, &mut y)?,
                                Some((e, scales)) => {
                                    let weight = E4m3Weight { qs: &qs, e, scales };
                                    if exact {
                                        dev.g32.enqueue_e4m3_tile(stream, weight, &x, n, &mut y)?
                                    } else {
                                        dev.g32.enqueue_e4m3_mma(stream, weight, &x, n, &mut y)?
                                    }
                                }
                            }
                            Ok::<_, GpuError>(y.to_host_vec(stream)?)
                        };
                        let (mma, exact) = (run(false)?, run(true)?);
                        // A BF16 weight's site over the columns' BF16 form: the same
                        // bits, and its BF16 store each cell's rounded to nearest even.
                        let (cols_bits, store_bits) = if scales.is_some() {
                            (true, None)
                        } else {
                            let mut act = GemmAct32::new(stream, n, k)?
                                .with_form(stream, bloomery_gpu::gemm::MmaForm::Bf16)?;
                            dev.g32.enqueue_bf16_cols(stream, &x, n, &mut act)?;
                            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n])?;
                            dev.g32
                                .enqueue_bf16_mma_cols(stream, &w, &x, &act, n, &mut y)?;
                            let got = y.to_host_vec(stream)?;
                            let mut half =
                                DeviceBuffer::from_host(stream, &vec![0x1234u16; rows * n])?;
                            dev.g32
                                .enqueue_bf16_mma_cols(stream, &w, &x, &act, n, &mut half)?;
                            let half = half.to_host_vec(stream)?;
                            (
                                got.iter()
                                    .zip(&mma)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                Some(
                                    half.iter()
                                        .zip(&mma)
                                        .all(|(h, v)| *h == gguf::quant::fp32_to_bf16_bits(*v)),
                                ),
                            )
                        };
                        let odd = |i: usize| {
                            (odd_r && i % rows == odd_row) || (odd_c && i / rows == odd_col)
                        };
                        let kept = (0..rows * n)
                            .filter(|&i| !odd(i))
                            .all(|i| mma[i].to_bits() == (k as f32).to_bits());
                        let replayed = (0..rows * n)
                            .filter(|&i| odd(i))
                            .all(|i| mma[i].to_bits() == exact[i].to_bits());
                        let pass = kept && replayed && cols_bits && store_bits != Some(false);
                        println!(
                            "dense_mma {format} cells rows={rows} K={k} n={n} row{odd_row}_odd={odd_r} \
                             col{odd_col}_odd={odd_c} others_kept_k={kept} odd_lines_exact={replayed} \
                             bf16_cols_bits={cols_bits} bf16_store_bits={} {}",
                            store_bits.map_or("n/a".into(), |b| b.to_string()),
                            verdict(pass)
                        );
                        ok &= pass;
                    }
                }
            }
            // The tile copies a BF16 weight's words in 16-byte pieces, so a
            // weight that does not start 16-byte aligned is refused by name.
            let words = DeviceBuffer::from_host(stream, &vec![0x3f80_3f80u32; 64 * 32 + 1])?;
            let w = DeviceTensor::<u32>::window_of(&words, 4, 64, 32)?;
            let x = DeviceBuffer::from_host(stream, &vec![1.0f32; 64])?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 64)?;
            let refused = matches!(
                gpu.q8f32()
                    .enqueue_bf16_mma(stream, &w, &x, 1, &mut y),
                Err(e) if e.to_string().contains("is not 16-byte aligned")
            );
            println!(
                "dense_mma bf16 weight_unaligned refused={refused} {}",
                verdict(refused)
            );
            ok &= refused;
            // A BF16 form is read only for the rows and the columns it was rounded from.
            let w = DeviceTensor::upload(stream, &vec![0x3f80_3f80u32; 64 * 32], 64, 32)?;
            let (one, two) = (
                DeviceBuffer::from_host(stream, &vec![1.0f32; 128])?,
                DeviceBuffer::from_host(stream, &vec![1.0f32; 128])?,
            );
            let mut act = GemmAct32::new(stream, 2, 64)?
                .with_form(stream, bloomery_gpu::gemm::MmaForm::Bf16)?;
            dev.g32.enqueue_bf16_cols(stream, &one, 1, &mut act)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 128)?;
            let stale = |r: Result<(), GpuError>| matches!(r, Err(e) if e.to_string().contains("BF16 form"));
            let other_rows = stale(
                dev.g32
                    .enqueue_bf16_mma_cols(stream, &w, &two, &act, 1, &mut y),
            );
            let more_cols = stale(
                dev.g32
                    .enqueue_bf16_mma_cols(stream, &w, &one, &act, 2, &mut y),
            );
            let fresh = dev
                .g32
                .enqueue_bf16_mma_cols(stream, &w, &one, &act, 1, &mut y)
                .is_ok();
            dev.g32
                .enqueue_quantize_gemm32(stream, &two, 1, &mut act, gpu.unlabelled_sink())?;
            let requantized = stale(
                dev.g32
                    .enqueue_bf16_mma_cols(stream, &w, &one, &act, 1, &mut y),
            );
            let pass = other_rows && more_cols && fresh && requantized;
            println!(
                "dense_mma bf16_cols stale_refused other_rows={other_rows} more_cols={more_cols} \
                 fresh_read={fresh} after_quantize={requantized} {}",
                verdict(pass)
            );
            ok &= pass;
            Ok(ok)
        }

        /// `enqueue_mxfp8_cols`'s values of one column, on the host: each
        /// 32-value block's scale `2^(exp - 127)`, `exp` the biased exponent of
        /// `amax · (1 / 448)` rounded up and held to 1..=254, and each value
        /// over that scale as E4M3 (nearest, ties to even, saturated at 448)
        /// times the scale.
        fn mxfp8_reference(col: &[f32]) -> Vec<f32> {
            let e4m3 = |v: f64| {
                let a = v.abs().min(448.0);
                let exp = ((a.to_bits() >> 52) as i32 - 1023).max(-6);
                let step = 2.0f64.powi(exp - 3);
                ((a / step).round_ties_even() * step).min(448.0).copysign(v)
            };
            col.as_chunks::<32>()
                .0
                .iter()
                .flat_map(|block| {
                    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let bits = (amax * (1.0 / 448.0)).to_bits();
                    let exp = ((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)).clamp(1, 254);
                    let scale = 2.0f64.powi(exp as i32 - 127);
                    block
                        .iter()
                        .map(move |&v| (e4m3(f64::from(v) / scale) * scale) as f32)
                })
                .collect()
        }

        /// The gated norm writing the next site's q8 form in its own launch
        /// against the norm and then the form's quantizer: the rows, the
        /// q8 blocks of 32 and the q8_1 blocks of 128 bit-equal under both
        /// gates, at one token, a ragged count and a ubatch, a head holding
        /// a NaN refused alike with the fault raised on both sides, and a
        /// scratch holding the native FP4 form refused untouched.
        fn norm_form_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            use bloomery_gpu::linear::norm_gate::{
                GdnGate, NormGateArgs, NormGateKernels, NormOut,
            };
            if !wanted("norm_form") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let norm = NormGateKernels::load(gpu.context())?;
            let n_v = 32usize;
            let k = 128 * n_v;
            let mut ok = true;
            for gate in [GdnGate::Silu, GdnGate::Sigmoid] {
                for (m, nan) in [(1usize, false), (9, false), (512, false), (9, true)] {
                    let mut o = activations(k, m, 4401 + m as u32);
                    if nan {
                        o[k + 3 * 128 + 5] = f32::NAN;
                    }
                    let o = DeviceBuffer::from_host(stream, &o)?;
                    let z = DeviceBuffer::from_host(stream, &activations(k, m, 4501))?;
                    let w = DeviceBuffer::from_host(stream, &activations(128, 1, 4601))?;
                    macro_rules! args {
                        ($y:expr) => {
                            NormGateArgs {
                                o: &o,
                                z: &z,
                                w: &w,
                                gate,
                                eps: 1e-6,
                                n_v,
                                m,
                                fault: sink,
                                y: $y,
                            }
                        };
                    }
                    let mut y_ref = DeviceBuffer::<f32>::zeroed(stream, m * k)?;
                    let mut r32 = GemmAct32::new(stream, m, k)?;
                    let mut r128 = GemmAct::new(stream, m, k)?;
                    norm.enqueue_norm_gate(stream, args!(&mut y_ref))?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &y_ref, m, &mut r32, sink)?;
                    gpu.enqueue_quantize_gemm(&y_ref, m, &mut r128, sink)?;
                    let ref_fault = gpu.take_fault()?.is_some();
                    let mut y32 = DeviceBuffer::<f32>::zeroed(stream, m * k)?;
                    let mut a32 = GemmAct32::new(stream, m, k)?;
                    norm.enqueue_norm_gate_into(stream, args!(&mut y32), NormOut::Q32(&mut a32))?;
                    let fault32 = gpu.take_fault()?.is_some();
                    let mut y128 = DeviceBuffer::<f32>::zeroed(stream, m * k)?;
                    let mut a128 = GemmAct::new(stream, m, k)?;
                    norm.enqueue_norm_gate_into(
                        stream,
                        args!(&mut y128),
                        NormOut::Q128(&mut a128),
                    )?;
                    let fault128 = gpu.take_fault()?.is_some();
                    let y = y_ref.to_host_vec(stream)?;
                    let rows_eq = bits_equal(&y32.to_host_vec(stream)?, &y)
                        && bits_equal(&y128.to_host_vec(stream)?, &y);
                    let q32_eq = a32.filled() == m
                        && a32.q().to_host_vec(stream)? == r32.q().to_host_vec(stream)?
                        && bits_equal(&a32.d().to_host_vec(stream)?, &r32.d().to_host_vec(stream)?)
                        && a32.s().to_host_vec(stream)? == r32.s().to_host_vec(stream)?;
                    let q128_eq = a128.q3().to_host_vec(stream)?
                        == r128.q3().to_host_vec(stream)?
                        && a128.q4().to_host_vec(stream)? == r128.q4().to_host_vec(stream)?
                        && a128.q6().to_host_vec(stream)? == r128.q6().to_host_vec(stream)?
                        && a128.s8().to_host_vec(stream)? == r128.s8().to_host_vec(stream)?
                        && bits_equal(
                            &a128.d8().to_host_vec(stream)?,
                            &r128.d8().to_host_vec(stream)?,
                        );
                    let fault_ok = ref_fault == nan && fault32 == nan && fault128 == nan;
                    let pass = rows_eq && q32_eq && q128_eq && fault_ok;
                    println!(
                        "gemm32 case=norm_form gate={gate:?} m={m} n_v={n_v} nan={nan} rows_eq={rows_eq} \
                         q32_eq_quantizer={q32_eq} q128_eq_quantizer={q128_eq} fault_ok={fault_ok} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // A routed scratch holding the native FP4 form is refused by name
            // before any write: the rows, the Q8 planes and the filled
            // columns stay as they were.
            let m = 9usize;
            let mut native = GemmAct32::new_routed(stream, m, k)?;
            if native.has_native_form() {
                let x = DeviceBuffer::from_host(stream, &activations(k, m, 4701))?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &x, m, &mut native, sink)?;
                let (q, d, s) = (
                    native.q().to_host_vec(stream)?,
                    native.d().to_host_vec(stream)?,
                    native.s().to_host_vec(stream)?,
                );
                let o = DeviceBuffer::from_host(stream, &activations(k, m, 4401))?;
                let z = DeviceBuffer::from_host(stream, &activations(k, m, 4501))?;
                let w = DeviceBuffer::from_host(stream, &activations(128, 1, 4601))?;
                let sentinel = vec![7.0f32; m * k];
                let mut y = DeviceBuffer::from_host(stream, &sentinel)?;
                let args = NormGateArgs {
                    o: &o,
                    z: &z,
                    w: &w,
                    gate: GdnGate::Sigmoid,
                    eps: 1e-6,
                    n_v,
                    m,
                    fault: sink,
                    y: &mut y,
                };
                let refused = matches!(
                    norm.enqueue_norm_gate_into(stream, args, NormOut::Q32(&mut native)),
                    Err(e) if e.to_string().contains("does not update the native FP4 form")
                );
                let kept = bits_equal(&y.to_host_vec(stream)?, &sentinel)
                    && native.filled() == m
                    && native.q().to_host_vec(stream)? == q
                    && bits_equal(&native.d().to_host_vec(stream)?, &d)
                    && native.s().to_host_vec(stream)? == s;
                let pass = refused && kept;
                println!(
                    "gemm32 case=norm_form native=true m={m} n_v={n_v} refused={refused} kept={kept} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The native W8A8 MXFP8 product over the activations' MXFP8 form:
        /// the form's values, block scales on both operands, tile tails,
        /// eager against graph, the cells the exact owner replays, and the
        /// stale and misplaced forms refused by name.
        fn mxfp8_w8a8_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("dense_mma_mxfp8_w8a8") {
                return Ok(true);
            }
            use bloomery_gpu::gemm::MmaForm;
            use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight};
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let g32 = &dev.g32;
            fn mxfp8<'a>(qs: &'a DeviceTensor<u32>, e: &'a DeviceTensor<u32>) -> E4m3Weight<'a> {
                E4m3Weight {
                    qs,
                    e,
                    scales: E4m3Scales::Mxfp8,
                }
            }
            let bundles = bloomery_gpu_gates::ptx::current_exe_bundles()?;
            let native = bloomery_gpu_gates::ptx::modules(&bundles)?.iter().any(|m| {
                bloomery_gpu_gates::ptx::body(std::slice::from_ref(m), "mxfp8_w8a8_gemm").is_some()
            });
            if !native {
                if super::case_filter().as_deref() == Some("mxfp8_w8a8") {
                    return Err("mxfp8_w8a8 requires the native block-scaled build".into());
                }
                // Without the native kernels the form is refused by name.
                let x = DeviceBuffer::from_host(stream, &[1.0f32; 128])?;
                let mut act = GemmAct32::new(stream, 1, 128)?.with_form(stream, MmaForm::Mxfp8)?;
                let refused = matches!(
                    g32.enqueue_mxfp8_cols(stream, &x, 1, &mut act),
                    Err(e) if e.to_string().contains("native block-scaled build")
                );
                println!(
                    "gemm32 case=mxfp8_w8a8 native=false refused={refused} {}",
                    verdict(refused)
                );
                return Ok(refused);
            }
            let mut ok = true;
            // Unit weights pick one activation each: every cell is one value of
            // the form, against the host's quantizer, over blocks of every
            // magnitude, rounding ties, E4M3 subnormals, zero blocks and amax on
            // a power of two times 448.
            {
                let (k, n) = (256usize, 133usize);
                let mut rng = bloomery_gpu_gates::GateRng(0x6d78_3861);
                let mut xh = vec![0.0f32; n * k];
                for (b, block) in xh.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                    let mag = 2.0f32.powi(rng.below(41) as i32 - 20);
                    match b % 9 {
                        0 => {}
                        1 => {
                            let ties = [1.0625f32, 1.1875, 17.0, -17.0, 0.75 / 512.0, 0.25 / 512.0];
                            for (z, v) in block.iter_mut().enumerate() {
                                *v = mag * ties[z % ties.len()];
                            }
                            block[31] = mag * 448.0;
                        }
                        _ => {
                            for v in block.iter_mut() {
                                *v =
                                    mag * rng.uni(-1.0, 1.0) * 2.0f32.powi(-(rng.below(13) as i32));
                            }
                        }
                    }
                }
                let rows = k;
                let qh: Vec<u32> = (0..rows * k / 4)
                    .map(|w| {
                        let (r, z) = (4 * w / k, 4 * w % k);
                        if r / 4 == z / 4 {
                            0x38 << (8 * (r % 4))
                        } else {
                            0
                        }
                    })
                    .collect();
                let qs = DeviceTensor::upload(stream, &qh, rows, k / 4)?;
                let e = DeviceTensor::upload(
                    stream,
                    &vec![0x7f7f_7f7fu32; rows * k / 128],
                    rows,
                    k / 128,
                )?;
                let x = DeviceBuffer::from_host(stream, &xh)?;
                let mut act = GemmAct32::new(stream, n, k)?.with_form(stream, MmaForm::Mxfp8)?;
                let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n])?;
                g32.enqueue_mxfp8_cols(stream, &x, n, &mut act)?;
                g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &x, &act, n, &mut y)?;
                let got = y.to_host_vec(stream)?;
                let want: Vec<f32> = xh.chunks_exact(k).flat_map(mxfp8_reference).collect();
                let bad = (0..rows * n).filter(|&at| got[at] != want[at]).count();
                println!(
                    "gemm32 case=mxfp8_w8a8_form K={k} n={n} cells_off_reference={bad} {}",
                    verdict(bad == 0)
                );
                ok &= bad == 0;
            }
            for (rows, k, n) in [
                (129, 128, 127),
                (129, 128, 128),
                (129, 256, 129),
                (257, 128, 257),
                (128, 640, 129),
                (17, 640, 3),
            ] {
                // Weights of three binary digits on per-block scales 2^-3 to
                // 2^3, activations whose E4M3 values carry three: every
                // product and partial sum is exact in f32.
                let mut qh = vec![0u32; rows * k / 4];
                let mut eh = vec![0u32; rows * k / 128];
                let mut wh = vec![0.0f32; rows * k];
                for r in 0..rows {
                    for b in 0..k / 32 {
                        let exp = 124 + (r + b) % 7;
                        eh[r * k / 128 + b / 4] |= (exp as u32) << (8 * (b % 4));
                        for z in 0..32 {
                            let (code, value) = [
                                (0x38u32, 1.0f32),
                                (0xb8, -1.0),
                                (0x30, 0.5),
                                (0x40, 2.0),
                                (0x40, 2.0),
                                (0x30, 0.5),
                                (0xb8, -1.0),
                                (0x38, 1.0),
                            ][(z + r + z / 8) % 8];
                            let at = r * k + 32 * b + z;
                            qh[at / 4] |= code << (8 * (at % 4));
                            wh[at] = value * 2.0f32.powi(exp as i32 - 127);
                        }
                    }
                }
                let mut xh = vec![0.0f32; n * k];
                for c in 0..n {
                    for b in 0..k / 32 {
                        let scale = 2.0f32.powi(((c + b) % 5) as i32 - 2);
                        for z in 0..32 {
                            let j = (z + c + b) % 8;
                            xh[c * k + b * 32 + z] =
                                [0.0, 1.0625, 1.1875, 1.5, 1.75, -1.0625, -1.1875, -1.5][j]
                                    * scale
                                    * [1.0, 0.5, 0.25, 0.125][z / 8];
                        }
                    }
                }
                let xq: Vec<f32> = xh.chunks_exact(k).flat_map(mxfp8_reference).collect();
                let qs = DeviceTensor::upload(stream, &qh, rows, k / 4)?;
                let e = DeviceTensor::upload(stream, &eh, rows, k / 128)?;
                let x = DeviceBuffer::from_host(stream, &xh)?;
                let mut act = GemmAct32::new(stream, n, k)?.with_form(stream, MmaForm::Mxfp8)?;
                let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 16])?;
                let mut enqueue = |s: &cuda_core::CudaStream,
                                   x: &DeviceBuffer<f32>,
                                   e: &DeviceTensor<u32>,
                                   y: &mut DeviceBuffer<f32>| {
                    g32.enqueue_mxfp8_cols(s, x, n, &mut act)?;
                    g32.enqueue_mxfp8_w8a8(s, mxfp8(&qs, e), x, &act, n, y)
                };
                enqueue(stream, &x, &e, &mut y)?;
                let eager = y.to_host_vec(stream)?;
                // The replay runs over a form rewritten from zeros after the
                // capture, so its own form launch counts.
                let graph = gpu.capture(|s| enqueue(s, &x, &e, &mut y))?;
                enqueue(
                    stream,
                    &DeviceBuffer::from_host(stream, &vec![0.0f32; n * k])?,
                    &e,
                    &mut y,
                )?;
                y.copy_from_host(stream, &vec![SENT; y.len()])?;
                graph.launch(stream)?;
                let replay = y.to_host_vec(stream)?;
                let mut off = 0usize;
                for c in 0..n {
                    for r in 0..rows {
                        let want = (0..k)
                            .map(|z| f64::from(wh[r * k + z]) * f64::from(xq[c * k + z]))
                            .sum::<f64>() as f32;
                        off += usize::from(eager[c * rows + r] != want);
                    }
                }
                let graph_equal = bits_equal(&eager, &replay);
                let guard = eager[rows * n..].iter().all(|v| *v == SENT);
                // The BF16 store: each cell's F32 bits rounded to nearest even,
                // a NaN as a NaN.
                let same16 =
                    |a: u16, b: u16| a == b || (a & 0x7fff > 0x7f80 && b & 0x7fff > 0x7f80);
                let stored16 = |x: &DeviceBuffer<f32>, e: &DeviceTensor<u32>, want: &[f32]| {
                    let mut half =
                        DeviceBuffer::from_host(stream, &vec![0x1234u16; rows * n + 16])?;
                    let mut a = GemmAct32::new(stream, n, k)?.with_form(stream, MmaForm::Mxfp8)?;
                    g32.enqueue_mxfp8_cols(stream, x, n, &mut a)?;
                    g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, e), x, &a, n, &mut half)?;
                    let half = half.to_host_vec(stream)?;
                    Ok::<_, GateError>(
                        (0..rows * n)
                            .all(|at| same16(half[at], gguf::quant::fp32_to_bf16_bits(want[at])))
                            && half[rows * n..].iter().all(|&h| h == 0x1234),
                    )
                };
                let bf16 = stored16(&x, &e, &eager)?;
                let pass = off == 0 && graph_equal && guard && bf16;
                println!(
                    "gemm32 case=mxfp8_w8a8 rows={rows} K={k} n={n} cells_off_oracle={off} eager_graph_eq={graph_equal} guard={guard} bf16_store_bits={bf16} {}",
                    verdict(pass)
                );
                ok &= pass;
                // A value the tensor cores cannot take sends its column to
                // the exact owner; a scale byte out of 10..=165 its row.
                // Every other cell keeps its W8A8 bits.
                let exact_of = |x: &DeviceBuffer<f32>, e: &DeviceTensor<u32>| {
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n])?;
                    g32.enqueue_e4m3_tile(stream, mxfp8(&qs, e), x, n, &mut y)?;
                    Ok::<_, GateError>(y.to_host_vec(stream)?)
                };
                let same =
                    |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
                let c = n.min(64) - 1;
                for odd in [f32::from_bits(0x200), f32::from_bits(0x5780_0000), f32::NAN] {
                    let mut dirty = xh.clone();
                    dirty[c * k + 1] = odd;
                    let dirty = DeviceBuffer::from_host(stream, &dirty)?;
                    enqueue(stream, &dirty, &e, &mut y)?;
                    let got = y.to_host_vec(stream)?;
                    let exact = exact_of(&dirty, &e)?;
                    let cells = (0..rows * n).all(|at| {
                        same(got[at], if at / rows == c { exact[at] } else { eager[at] })
                    });
                    let bf16 = stored16(&dirty, &e, &got)?;
                    println!(
                        "gemm32 case=mxfp8_w8a8_neighbours rows={rows} K={k} n={n} col={c} odd={:08x} cells={cells} bf16_store_bits={bf16} {}",
                        odd.to_bits(),
                        verdict(cells && bf16)
                    );
                    ok &= cells && bf16;
                }
                let mut edges = eh.clone();
                let per_row = k / 128;
                for (r, byte) in [(0, 9u32), (1, 166), (2, 10), (3, 165)] {
                    edges[r * per_row] = (edges[r * per_row] & !255) | byte;
                }
                let edges = DeviceTensor::upload(stream, &edges, rows, per_row)?;
                enqueue(stream, &x, &edges, &mut y)?;
                let got = y.to_host_vec(stream)?;
                let exact = exact_of(&x, &edges)?;
                let line = |r: usize| (0..n).map(move |c| c * rows + r);
                let replayed = line(0).chain(line(1)).all(|at| same(got[at], exact[at]));
                let kept = [2, 3]
                    .iter()
                    .all(|&r| line(r).any(|at| !same(got[at], exact[at])));
                let others = (0..rows * n)
                    .filter(|at| at % rows > 3)
                    .all(|at| same(got[at], eager[at]));
                let pass = replayed && kept && others;
                println!(
                    "gemm32 case=mxfp8_w8a8_scale_edges rows={rows} K={k} n={n} bytes_9_166_exact={replayed} \
                     bytes_10_165_mma={kept} others_kept={others} {}",
                    verdict(pass)
                );
                ok &= pass;
                // One refused scale byte alone, in the last block of its word:
                // row 8 (a thread's first copy line), row 64 (a second copy
                // line) at the last word, row 5 at the last word. Only that row
                // replays.
                for (r, word) in [(8, 0), (64, per_row - 1), (5, per_row - 1)] {
                    if r >= rows {
                        continue;
                    }
                    let mut one = eh.clone();
                    one[r * per_row + word] = (one[r * per_row + word] & 0x00ff_ffff) | (9 << 24);
                    let one = DeviceTensor::upload(stream, &one, rows, per_row)?;
                    enqueue(stream, &x, &one, &mut y)?;
                    let got = y.to_host_vec(stream)?;
                    let want = exact_of(&x, &one)?;
                    let replayed = line(r).all(|at| same(got[at], want[at]));
                    let others = (0..rows * n)
                        .filter(|at| at % rows != r)
                        .all(|at| same(got[at], eager[at]));
                    let pass = replayed && others;
                    println!(
                        "gemm32 case=mxfp8_w8a8_one_refusal rows={rows} K={k} n={n} row={r} word={word} \
                         replayed={replayed} others_kept={others} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // The form is read only for the rows and the columns it was made
            // from, until the next quantizer launch; a weight's planes
            // are read 16-byte aligned; an MXFP8 form takes K a multiple of 128.
            let (rows, k) = (64usize, 128usize);
            let qs =
                DeviceTensor::upload(stream, &vec![0x3838_3838u32; rows * k / 4], rows, k / 4)?;
            let e =
                DeviceTensor::upload(stream, &vec![0x7f7f_7f7fu32; rows * k / 128], rows, k / 128)?;
            let (one, two) = (
                DeviceBuffer::from_host(stream, &vec![1.0f32; 2 * k])?,
                DeviceBuffer::from_host(stream, &vec![1.0f32; 2 * k])?,
            );
            let mut act = GemmAct32::new(stream, 2, k)?.with_form(stream, MmaForm::Mxfp8)?;
            g32.enqueue_mxfp8_cols(stream, &one, 1, &mut act)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 2 * rows)?;
            let stale = |r: Result<(), GpuError>| matches!(r, Err(e) if e.to_string().contains("MXFP8 form"));
            let other_rows =
                stale(g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &two, &act, 1, &mut y));
            let more_cols =
                stale(g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &one, &act, 2, &mut y));
            let bf16_only = {
                let mut bf16 = GemmAct32::new(stream, 1, k)?.with_form(stream, MmaForm::Bf16)?;
                g32.enqueue_bf16_cols(stream, &one, 1, &mut bf16)?;
                stale(g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &one, &bf16, 1, &mut y))
            };
            let fresh = g32
                .enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &one, &act, 1, &mut y)
                .is_ok();
            g32.enqueue_quantize_gemm32(stream, &two, 1, &mut act, gpu.unlabelled_sink())?;
            let requantized =
                stale(g32.enqueue_mxfp8_w8a8(stream, mxfp8(&qs, &e), &one, &act, 1, &mut y));
            let words = DeviceBuffer::from_host(stream, &vec![0x3838_3838u32; rows * k / 4 + 1])?;
            let shifted = DeviceTensor::<u32>::window_of(&words, 4, rows, k / 4)?;
            g32.enqueue_mxfp8_cols(stream, &one, 1, &mut act)?;
            let unaligned = matches!(
                g32.enqueue_mxfp8_w8a8(stream, mxfp8(&shifted, &e), &one, &act, 1, &mut y),
                Err(e) if e.to_string().contains("is not 16-byte aligned")
            );
            let short_k = matches!(
                GemmAct32::new(stream, 1, 64)?.with_form(stream, MmaForm::Mxfp8),
                Err(e) if e.to_string().contains("multiple of 128")
            );
            let pass = other_rows
                && more_cols
                && bf16_only
                && fresh
                && requantized
                && unaligned
                && short_k;
            println!(
                "gemm32 case=mxfp8_w8a8_refused other_rows={other_rows} more_cols={more_cols} \
                 bf16_form_only={bf16_only} fresh_read={fresh} after_quantize={requantized} \
                 unaligned_weight={unaligned} k_not_128={short_k} {}",
                verdict(pass)
            );
            ok &= pass;
            Ok(ok)
        }

        /// The input of one `nvfp4_native` case.
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Native {
            /// Powers of two, exact on both formats' grids.
            Pattern,
            /// One signed value a column.
            OneHot,
            /// The pattern at subnormal magnitude.
            Tiny,
            /// Values off the E2M1 grid, held to the host W4A4 transcription.
            Varied,
            /// A NaN among zeros, a NaN beside varied values, and a varied column:
            /// `QuantColumn`, and NaN in every row of a refused column.
            Refused,
            /// One expert scale and one input value at the edge of the range.
            Range(f32, f32),
        }

        fn nvfp4_native_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("nvfp4_native") {
                return Ok(true);
            }
            let native = GemmAct32::new_routed(dev.gpu.stream(), 1, 64)?.bytes()
                > GemmAct32::new(dev.gpu.stream(), 1, 64)?.bytes();
            if !native {
                return if super::case_filter().as_deref() == Some("nvfp4_native") {
                    Err("nvfp4_native requires the native FP4 build".into())
                } else {
                    Ok(true)
                };
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let mut ok = true;
            // Each input 4 bytes past a 16-byte boundary: refused by name before the launch.
            let parent = DeviceBuffer::from_host(stream, &[1.0f32; 65])?;
            let whole = bloomery_gpu::Window::<f32>::of(&parent, 0, 64)?;
            let off = bloomery_gpu::Window::<f32>::of(&parent, 4, 64)?;
            for (name, g, u) in [("g", &off, &whole), ("u", &whole, &off)] {
                let mut act = GemmAct32::new_routed(stream, 1, 64)?;
                let got = dev.g32.enqueue_swiglu_quant32(
                    stream,
                    g,
                    u,
                    1,
                    &mut act,
                    gpu.unlabelled_sink(),
                );
                let refused = matches!(
                    &got,
                    Err(bloomery_gpu::GpuError::Shape { what, detail })
                        if *what == "NVFP4 activation"
                            && detail.starts_with(&format!("{name} at "))
                            && detail.contains("is not 16-byte aligned")
                );
                let pass = refused && gpu.take_fault()?.is_none();
                println!(
                    "gemm32 case=nvfp4_native misaligned={name} refused={refused} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            for (rows, slots, k, input) in [
                (16usize, 14usize, 128usize, Native::Pattern),
                (128, 73, 64, Native::Pattern),
                (144, 129, 256, Native::Pattern),
                (128, 129, 256, Native::Pattern),
                // live_walk's 1, 2, 4 and 6 fragment arms, varied per slot and K so an
                // activation row the walk fails to copy cannot match a stale one.
                (128, 14, 128, Native::Varied),
                (128, 25, 128, Native::Varied),
                (128, 49, 128, Native::Varied),
                (128, 73, 128, Native::Varied),
                (48, 73, 64, Native::OneHot),
                (16, 14, 20_480, Native::Pattern),
                (16, 14, 64, Native::Tiny),
                (16, 14, 64, Native::Varied),
                (16, 14, 192, Native::Varied),
                (16, 14, 448, Native::Varied),
                (16, 3, 64, Native::Refused),
                (
                    16,
                    1,
                    64,
                    Native::Range(2.0f32.powi(120), 2.0f32.powi(-100)),
                ),
                (16, 1, 64, Native::Range(2.0f32.powi(-120), f32::MAX)),
                (16, 1, 64, Native::Range(2.0f32.powi(-120), -f32::MAX)),
                (
                    16,
                    1,
                    64,
                    Native::Range(2.0f32.powi(-100), 2.0f32.powi(120)),
                ),
                (
                    16,
                    1,
                    64,
                    Native::Range(f32::from_bits(1), 2.0f32.powi(120)),
                ),
                (
                    16,
                    1,
                    64,
                    Native::Range(f32::from_bits(0x7b2aa555), f32::from_bits(0x40c00600)),
                ),
            ] {
                let range = match input {
                    Native::Range(scale, x) => Some((scale, x)),
                    _ => None,
                };
                let (one_hot, tiny) = (input == Native::OneHot, input == Native::Tiny);
                let refused = input == Native::Refused;
                // Off-grid inputs meet the host W4A4 transcription and weights that vary along K.
                let host = refused || input == Native::Varied;
                let block_scale = if range.is_some_and(|(s, _)| s == f32::from_bits(1)) {
                    0.001953125f32
                } else {
                    1.0
                };
                let codes: Vec<u32> = (0..2 * rows * k / 8)
                    .map(|i| {
                        let row = i / (k / 8) % rows;
                        if range.is_some() {
                            return 0x22222222;
                        }
                        if !(one_hot || host) {
                            return (2 + row as u32 % 3) * 0x11111111;
                        }
                        let word = i % (k / 8);
                        let at = 16 * (word / 2) + 4 * (word & 1);
                        (0..4).fold(0u32, |bits, j| {
                            let lo = ((row + 3 * (at + j)) & 15) as u32;
                            let hi = ((row + 3 * (at + j + 8)) & 15) as u32;
                            bits | (lo | (hi << 4)) << (8 * j)
                        })
                    })
                    .collect();
                let wq = DeviceTensor::upload(stream, &codes, 2 * rows, k / 8)?;
                let we = DeviceTensor::upload(
                    stream,
                    &vec![
                        if range.is_some() {
                            if block_scale == 1.0 {
                                0x38383838u32
                            } else {
                                0x01010101u32
                            }
                        } else {
                            0x50483840u32
                        };
                        2 * rows * k / 64
                    ],
                    2 * rows,
                    k / 64,
                )?;
                let weight_scales = range.map_or([1.0f32, 0.5], |(scale, _)| [scale; 2]);
                let ws = DeviceTensor::upload(stream, &weight_scales, 2, 1)?;
                let xs: Vec<f32> = (0..slots * k)
                    .map(|i| {
                        if let Some((_, input)) = range {
                            return input;
                        }
                        if one_hot {
                            return if i % k == i / k % k {
                                [1.0, -1.0][i / k % 2]
                            } else {
                                0.0
                            };
                        }
                        if refused && (i == 0 || i == k + 5) {
                            return f32::NAN;
                        }
                        if refused && i < k {
                            return 0.0;
                        }
                        if host {
                            return ((i * 37 % 257) as f32 - 128.0) / 64.0;
                        }
                        let x =
                            2.0f32.powi((i / k % 4) as i32 - 2) * [2.0, 1.0, 4.0, 8.0][i / 16 % 4];
                        if tiny {
                            x * 1.0e-40 * [1.0, 0.5][i & 1]
                        } else {
                            x
                        }
                    })
                    .collect();
                let x_host: Vec<f32> = xs.iter().map(|v| (v / 128.0) * 128.0).collect();
                let x = DeviceBuffer::from_host(stream, &x_host)?;
                let want_fault = refused.then(|| Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let ids: Vec<u32> = (0..slots).map(|i| (i % 2) as u32).collect();
                let ids_d = DeviceBuffer::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, slots, 2)?;
                dev.gk
                    .enqueue_route(stream, &ids_d, slots, &mut route, gpu.unlabelled_sink())?;
                let g = DeviceBuffer::from_host(stream, &vec![128.0f32; slots * k])?;
                let u = DeviceBuffer::from_host(
                    stream,
                    &xs.iter().map(|v| v / 128.0).collect::<Vec<_>>(),
                )?;
                let mut plain = Vec::new();
                for (fused, native) in [(false, true), (true, true), (false, false)] {
                    if !native && range.is_none() && !refused {
                        continue;
                    }
                    let mut act = if native {
                        GemmAct32::new_routed(stream, slots, k)?
                    } else {
                        GemmAct32::new(stream, slots, k)?
                    };
                    let mut legacy = GemmAct32::new(stream, slots, k)?;
                    dev.g32.enqueue_swiglu_quant32_sel(
                        stream,
                        &g,
                        &u,
                        &ids_d,
                        2,
                        slots,
                        &mut legacy,
                        gpu.unlabelled_sink(),
                    )?;
                    let legacy_fault = gpu.take_fault()?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * rows + 16])?;
                    let graph = {
                        let mut enqueue = |s: &cuda_core::CudaStream| {
                            if fused {
                                dev.g32.enqueue_swiglu_quant32_sel(
                                    s,
                                    &g,
                                    &u,
                                    &ids_d,
                                    2,
                                    slots,
                                    &mut act,
                                    gpu.unlabelled_sink(),
                                )?;
                            } else {
                                dev.g32.enqueue_quantize_gemm32(
                                    s,
                                    &x,
                                    slots,
                                    &mut act,
                                    gpu.unlabelled_sink(),
                                )?;
                            }

                            dev.g32.enqueue_gemm32(
                                s,
                                Gemm32Args {
                                    w: Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                                        qs: &wq,
                                        e: &we,
                                        scales: &ws,
                                    }),
                                    rows_per_expert: rows,
                                    act: &act,
                                    route: &route,
                                    input: GemmInput::PerSlot,
                                    y: &mut y,
                                },
                            )
                        };
                        enqueue(stream)?;
                        gpu.capture(|s| enqueue(s))?
                    };
                    let q8_eq = !fused
                        || (act.q().to_host_vec(stream)? == legacy.q().to_host_vec(stream)?
                            && bits_equal(
                                &act.d().to_host_vec(stream)?,
                                &legacy.d().to_host_vec(stream)?,
                            )
                            && act.s().to_host_vec(stream)? == legacy.s().to_host_vec(stream)?);
                    let eager = y.to_host_vec(stream)?;
                    graph.launch(stream)?;
                    let replay = y.to_host_vec(stream)?;
                    let tail = eager[slots * rows..]
                        .iter()
                        .all(|v| v.to_bits() == SENT.to_bits());
                    // The BF16 product: each cell the F32 one rounded to nearest even, the
                    // tail untouched, on the native product and the Q8 one alike.
                    // PIN(2026-11-01): the Q8 NVFP4 arm has a BF16 store; its product and tail
                    // are checked where its refusal was.
                    let bf16_eq = fused
                        || bf16_product32(
                            dev,
                            Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                                qs: &wq,
                                e: &we,
                                scales: &ws,
                            }),
                            rows,
                            (&act, &route),
                            &eager[..slots * rows],
                        )?;
                    let host_want = if host {
                        let (ha, _) = host_act(&x_host, k, slots);
                        let host_w: Vec<i8> = (0..rows * k)
                            .map(|i| KVALUES_MXFP4[(i / k + 3 * (i % k)) & 15])
                            .collect();
                        let host_wsc: Vec<f32> = (0..k / 16)
                            .map(|s| ue4m3_half([0x40, 0x38, 0x48, 0x50][s % 4]))
                            .collect();
                        (0..slots * rows)
                            .map(|i| {
                                let (row, col) = (i % rows, i / rows);
                                let w = &host_w[row * k..(row + 1) * k];
                                let scale = weight_scales[ids[col] as usize];
                                if native {
                                    dot32_nv_fp4(w, &host_wsc, scale, &ha, col)
                                } else {
                                    let (_, want, mag) = dot32_nv(w, &host_wsc, scale, &ha, col);
                                    Ok((want, mag))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };
                    let wrong = eager[..slots * rows]
                        .iter()
                        .enumerate()
                        .filter(|(i, value)| {
                            if host {
                                let (want, mag) = host_want[*i];
                                // The block-scaled MMA sums at the hardware's own width, so
                                // the band is 2^-16 of the magnitude.
                                return if want.is_nan() {
                                    !value.is_nan()
                                } else {
                                    !value.is_finite()
                                        || (f64::from(**value) - want).abs()
                                            > mag * 2.0f64.powi(-16)
                                };
                            }
                            let row = i % rows;
                            let col = i / rows;
                            if let Some((scale, input)) = range {
                                let expected = (k as f64
                                    * f64::from(input)
                                    * f64::from(scale)
                                    * f64::from(block_scale))
                                    as f32;
                                return !value.is_finite()
                                    || (**value - expected).abs() > expected.abs() * 5.0e-7;
                            }
                            let dot = if one_hot {
                                let code = (row + 3 * (col % k)) & 15;
                                let level = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][code & 7];
                                let value = if code & 8 == 0 { level } else { -level };
                                value * [2.0, 1.0, 4.0, 8.0][(col % k) / 16] * [1.0, -1.0][col % 2]
                            } else {
                                (k / 64) as f32
                                    * 1360.0
                                    * [1.0, 1.5, 2.0][row % 3]
                                    * 2.0f32.powi((col % 4) as i32 - 2)
                            };
                            let expected = dot * [1.0, 0.5][ids[col] as usize];
                            let (expected, band) = if tiny {
                                let expected = expected * 0.75e-40;
                                // Subnormal Q8 scales have coarser relative spacing.
                                (expected, expected.abs() * 0.02)
                            } else {
                                (expected, 5.0e-7 * expected.abs().max(1.0))
                            };
                            !value.is_finite() || (**value - expected).abs() > band
                        })
                        .count();
                    let replay_eq = bits_equal(&eager, &replay);
                    let producer_eq = !fused || bits_equal(&eager, &plain);
                    let hash = Fnv1a64::default().f32s(&eager).value();
                    if !fused {
                        plain = eager;
                    }
                    let pass = tail
                        && wrong == 0
                        && replay_eq
                        && q8_eq
                        && producer_eq
                        && bf16_eq
                        && legacy_fault == want_fault
                        && gpu.take_fault()? == want_fault;
                    println!(
                        "gemm32 case=nvfp4_native rows={rows} slots={slots} k={k} input={input:?} fused={fused} native={native} q8_eq={q8_eq} producer_eq={producer_eq} bf16_eq={bf16_eq} wrong={wrong} guard={tail} eager_graph_eq={replay_eq} hash={hash:016x} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// The `nvfp4_gate_up` case: `enqueue_gate_up_sel32`'s native launch
        /// against the gate and up GEMMs and `swiglu_quant32_sel` it replaces,
        /// over a remapped route with host slots, and with a refused id or a
        /// NaN token: each card slot's FP4 is the transcription of the split
        /// pair's Q8 codes (`gemm32::gate_up_fp4`: levels, scale bytes and
        /// column scale), the other slots' the reset's zero column; the down
        /// GEMM's output holds a sentinel, a NaN or a finite value in the same
        /// cells for both, each finite card cell within the block-scaled MMA's
        /// band of the host down over the fused FP4; the raised fault is
        /// equal, `g` and `u` keep their sentinel (the native launch ran), and
        /// a graph replay over a reset output and overwritten activations
        /// leaves the same output and fault; down activations of another K
        /// than the gate and up rows are refused by name. Three cases reach the
        /// FP4's range edges (`Values`).
        fn nvfp4_gate_up_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            /// A case's values: `Spread` gives each expert's first 16 gate and
            /// up rows the UE4M3 scale 2^7 and the rest 1, so a column's
            /// sub-blocks and steps lie about 14 binades apart, across the
            /// lowest scale field the FP4 keeps; `Tiny`
            /// scales the input by 2^-66, so the SwiGLU nears 2^-121 and its
            /// column scale 2^-130; `Edge` makes every card column's gate 128
            /// and up `f32::MAX / 128`, so the SwiGLU is `f32::MAX` and 127
            /// times its Q8 block scale overflows f32. An `Edge` card column's
            /// FP4 levels must all be nonzero and its down outputs finite and
            /// nonzero, apart from the transcription.
            #[derive(Clone, Copy, Debug, PartialEq)]
            enum Values {
                Plain,
                Spread,
                Tiny,
                Edge,
            }
            if !wanted("nvfp4_gate_up") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let native = GemmAct32::new_routed(stream, 1, 64)?.bytes()
                > GemmAct32::new(stream, 1, 64)?.bytes();
            if !native {
                return if super::case_filter().as_deref() == Some("nvfp4_gate_up") {
                    Err("nvfp4_gate_up requires the native FP4 build".into())
                } else {
                    Ok(true)
                };
            }
            let sink = gpu.unlabelled_sink();
            let (n_card, hidden, top_k) = (3usize, 64usize, 2usize);
            const EXPERT_SCALES: [f32; 3] = [0.5, 1.0, 0.25];
            // Id 1 is the host's; id 4 maps past the card's experts (refused).
            let map = [0u32, HOST, 1, 2, 7];
            let map_d = DeviceBuffer::from_host(stream, &map)?;
            let mut ok = true;
            use Values::{Edge, Plain, Spread, Tiny};
            for (rows, k, tokens, refused, poison, varied, values) in [
                (64usize, 64usize, 9usize, false, None, false, Plain),
                (128, 128, 50, true, None, false, Plain),
                (64, 192, 23, false, None, false, Plain),
                (64, 448, 23, false, None, false, Plain),
                (192, 256, 37, false, Some(5usize), false, Plain),
                (192, 256, 37, false, None, false, Spread),
                (192, 256, 37, false, None, false, Tiny),
                (64, 64, 9, false, None, false, Edge),
                (640, 2560, 131, false, None, false, Plain),
                // Varied columns force the gate/up walk's 1, 2, 4, 6 and 8 live-fragment arms.
                (128, 128, 8, false, None, true, Plain),
                (128, 128, 16, false, None, true, Plain),
                (128, 128, 32, false, None, true, Plain),
                (128, 128, 48, false, None, true, Plain),
                (128, 128, 64, false, None, true, Plain),
            ] {
                let slots = tokens * top_k;
                let mut lcg = Lcg(0x5eed ^ (rows * k + tokens) as u64);
                let ids: Vec<u32> = (0..tokens)
                    .flat_map(|_| {
                        if varied {
                            return [0, 1];
                        }
                        let a = lcg.below(4 + usize::from(refused)) as u32;
                        let b = (a + 1 + lcg.below(3) as u32) % (4 + u32::from(refused));
                        [a, b]
                    })
                    .collect();
                let sel: Vec<u32> = ids.iter().map(|&id| map[id as usize]).collect();
                let ids_d = DeviceBuffer::from_host(stream, &ids)?;
                let sel_d = DeviceBuffer::from_host(stream, &sel)?;
                // Each stack's E4M3 block scales: random between 2^-3 and 2^2, or
                // fixed for each expert's first 16 rows and for the rest.
                let stack = |n_rows: usize,
                             n_k: usize,
                             seed: u64,
                             bytes: Option<[u32; 2]>,
                             scales: [f32; 3]|
                 -> Result<_, GateError> {
                    let mut g = Lcg(seed);
                    // `Edge` weights hold one level-1 code in each eight values.
                    let codes: Vec<u32> = (0..n_card * n_rows * n_k / 8)
                        .map(|_| if values == Edge { 2 } else { g.next() as u32 })
                        .collect();
                    let e: Vec<u32> = (0..n_card * n_rows * n_k / 64)
                        .map(|i| {
                            let top = i / (n_k / 64) % n_rows < 16;
                            (0..4u32).fold(0u32, |w, j| {
                                let byte = match bytes {
                                    None => 0x20 + g.below(40) as u32,
                                    Some([first, rest]) => {
                                        if top {
                                            first
                                        } else {
                                            rest
                                        }
                                    }
                                };
                                w | (byte << (8 * j))
                            })
                        })
                        .collect();
                    Ok((
                        DeviceTensor::upload(stream, &codes, n_card * n_rows, n_k / 8)?,
                        DeviceTensor::upload(stream, &e, n_card * n_rows, n_k / 64)?,
                        DeviceTensor::upload(stream, &scales, n_card, 1)?,
                    ))
                };
                // `Edge`: the input's 2^-3 column scale times the level-6 input and
                // level-1 weight, 6, times these expert scales gives 128 and f32::MAX / 128.
                let (gate_up_bytes, gate_scales, up_scales, down_bytes) = match values {
                    Plain | Tiny => (None, EXPERT_SCALES, EXPERT_SCALES, None),
                    Spread => (Some([0x70, 0x38]), EXPERT_SCALES, EXPERT_SCALES, None),
                    Edge => (
                        Some([0x38, 0x38]),
                        [512.0 / 3.0; 3],
                        [f32::MAX / 96.0; 3],
                        Some([0x08, 0x08]),
                    ),
                };
                let (gq, ge, gs) = stack(rows, k, 1, gate_up_bytes, gate_scales)?;
                let (uq, ue, us) = stack(rows, k, 2, gate_up_bytes, up_scales)?;
                let (dq, de, ds) = stack(hidden, rows, 3, down_bytes, EXPERT_SCALES)?;
                let gate = Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                    qs: &gq,
                    e: &ge,
                    scales: &gs,
                });
                let up = Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                    qs: &uq,
                    e: &ue,
                    scales: &us,
                });
                let down = Gemm32Weight::Nvfp4Planes(Nvfp4Stack {
                    qs: &dq,
                    e: &de,
                    scales: &ds,
                });
                let mut lx = Lcg(7 + tokens as u64);
                let x_scale = if values == Tiny { 2f32.powi(-66) } else { 1.0 };
                let xs: Vec<f32> = (0..tokens * k)
                    .map(|i| {
                        if poison == Some(i / k) && i % k == 3 {
                            f32::NAN
                        } else if values == Edge {
                            // Its Q8 scale c / 127 times 127 / 6 rounds to 2^-3 in f32.
                            if i % k == 0 {
                                f32::from_bits(0x3f40_0001)
                            } else {
                                0.0
                            }
                        } else {
                            (lx.below(4001) as f32 - 2000.0) / 1000.0 * x_scale
                        }
                    })
                    .collect();
                let x_d = DeviceBuffer::from_host(stream, &xs)?;
                let mut x = GemmAct32::new_routed(stream, tokens, k)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &x_d, tokens, &mut x, sink)?;
                let x_fault = gpu.take_fault()?;
                let mut route = GemmRoute::new(stream, slots, n_card)?;
                let blank = DeviceBuffer::from_host(stream, &vec![0.0f32; slots * rows])?;
                let mut runs = Vec::new();
                for fused in [false, true] {
                    let scratch_cells = if fused { 0 } else { slots * rows };
                    let mut g = DeviceBuffer::from_host(stream, &vec![SENT; scratch_cells])?;
                    let mut u = DeviceBuffer::from_host(stream, &vec![SENT; scratch_cells])?;
                    let mut act = if fused {
                        GemmAct32::new_gate_up_output(stream, slots, rows)?
                    } else {
                        GemmAct32::new_routed(stream, slots, rows)?
                    };
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * hidden])?;
                    let mut enqueue = |s: &cuda_core::CudaStream,
                                       g: &mut DeviceBuffer<f32>,
                                       u: &mut DeviceBuffer<f32>,
                                       act: &mut GemmAct32,
                                       y: &mut DeviceBuffer<f32>|
                     -> Result<(), GpuError> {
                        dev.g32
                            .enqueue_route_remap(s, &ids_d, &map_d, slots, &mut route, sink)?;
                        if fused {
                            dev.g32.enqueue_gate_up_sel32(
                                s,
                                GateUp32Args {
                                    w: [gate, up],
                                    rows_per_expert: rows,
                                    x: &x,
                                    route: &route,
                                    input: GemmInput::Shared { top_k },
                                    sel: &sel_d,
                                    n_card,
                                    g,
                                    u,
                                    act,
                                },
                                sink,
                            )?;
                        } else {
                            for (w, y) in [(gate, &mut *g), (up, &mut *u)] {
                                dev.g32.enqueue_gemm32(
                                    s,
                                    Gemm32Args {
                                        w,
                                        rows_per_expert: rows,
                                        act: &x,
                                        route: &route,
                                        input: GemmInput::Shared { top_k },
                                        y,
                                    },
                                )?;
                            }
                            dev.g32.enqueue_swiglu_quant32_sel(
                                s, g, u, &sel_d, n_card, slots, act, sink,
                            )?;
                        }
                        dev.g32.enqueue_gemm32(
                            s,
                            Gemm32Args {
                                w: down,
                                rows_per_expert: hidden,
                                act,
                                route: &route,
                                input: GemmInput::PerSlot,
                                y,
                            },
                        )
                    };
                    enqueue(stream, &mut g, &mut u, &mut act, &mut y)?;
                    let fault = gpu.take_fault()?;
                    let eager = y.to_host_vec(stream)?;
                    let graph = gpu.capture(|s| enqueue(s, &mut g, &mut u, &mut act, &mut y))?;
                    // The replay must rewrite the eager run's activations and output.
                    y.copy_from_host(stream, &vec![SENT; slots * hidden])?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &blank, slots, &mut act, sink)?;
                    let reset_fault = gpu.take_fault()?;
                    graph.launch(stream)?;
                    let replay = y.to_host_vec(stream)?;
                    let replay_fault = gpu.take_fault()?;
                    let untouched = g
                        .to_host_vec(stream)?
                        .iter()
                        .chain(&u.to_host_vec(stream)?)
                        .all(|v| v.to_bits() == SENT.to_bits());
                    let (fq, fe, fs) = act.fp4_planes().ok_or("a native scratch")?;
                    runs.push((
                        eager,
                        replay,
                        fault,
                        untouched,
                        reset_fault.is_none() && replay_fault == fault,
                        (act.q().to_host_vec(stream)?, act.d().to_host_vec(stream)?),
                        (
                            fq.to_host_vec(stream)?,
                            fe.to_host_vec(stream)?,
                            fs.to_host_vec(stream)?,
                        ),
                        act.bytes(),
                    ));
                }
                let (plain, fused) = (&runs[0], &runs[1]);
                let steps = rows / 64;
                let ((q8, d8), (fq, fe, fs)) = (&plain.5, &fused.6);
                let mut fp4_off = 0usize;
                for (s, &place) in sel.iter().enumerate() {
                    let (qr, er) = (s * rows / 8..(s + 1) * rows / 8, s * steps..(s + 1) * steps);
                    if place as usize >= n_card {
                        // The reset's zero column, which the native launch leaves.
                        fp4_off += usize::from(
                            fq[qr].iter().chain(&fe[er]).any(|&w| w != 0) || fs[s] != 1.0,
                        );
                        continue;
                    }
                    let codes: Vec<i8> = q8[s * 16 * steps..(s + 1) * 16 * steps]
                        .iter()
                        .flat_map(|w| w.to_le_bytes().map(|b| b as i8))
                        .collect();
                    let want = gate_up_fp4(&codes, &d8[s * 2 * steps..(s + 1) * 2 * steps]);
                    let got = fq[qr].iter().enumerate().flat_map(|(w, &word)| {
                        // Byte i of a sub-block's word pair holds values i (low nibble) and i + 8.
                        (0..4).flat_map(move |b| {
                            let at = 16 * (w / 2) + 4 * (w % 2) + b;
                            let byte = (word >> (8 * b)) & 255;
                            [(at, byte & 15), (at + 8, byte >> 4)]
                        })
                    });
                    fp4_off += usize::from(
                        got.into_iter().any(|(at, n)| {
                            f32::from(KVALUES_MXFP4[n as usize]) / 2.0 != want.levels[at]
                        }) || fe[er] != want.scale_words[..]
                            || !bits_equal(&[fs[s]], &[want.scale]),
                    );
                }
                // Each cell a sentinel, a NaN or finite in both runs.
                let kind = |v: f32| (v.to_bits() == SENT.to_bits(), v.is_nan());
                let y_kind_eq = plain
                    .0
                    .iter()
                    .zip(&fused.0)
                    .all(|(&a, &b)| kind(a) == kind(b));
                // PIN(2026-10-31): power-of-two FP4 frames replace exact split-output equality.
                // The down over the fused FP4 on the host: each card cell within 2^-16 of its
                // magnitude (the block-scaled MMA sums at its own width) plus half the f32
                // spacing below 2^-126 the store rounds to, NaN where its column is refused.
                let (dqh, deh) = (dq.buf().to_host_vec(stream)?, de.buf().to_host_vec(stream)?);
                let code = |words: &[u32], v: usize| {
                    let byte = 8 * (v / 16) + v % 8;
                    let n = (words[byte / 4] >> (8 * (byte % 4) + 4 * (v % 16 / 8))) & 15;
                    i32::from(KVALUES_MXFP4[n as usize])
                };
                let half = |words: &[u32], sb: usize| {
                    f64::from(ue4m3_half((words[sb / 4] >> (8 * (sb % 4))) & 255))
                };
                let mut y_off = 0usize;
                for (i, &value) in fused.0.iter().enumerate() {
                    let (s, row) = (i / hidden, i % hidden);
                    let place = sel[s] as usize;
                    if place >= n_card {
                        continue;
                    }
                    let w0 = place * hidden + row;
                    let (w, we) = (&dqh[w0 * rows / 8..], &deh[w0 * rows / 64..]);
                    let (a, ae) = (&fq[s * rows / 8..], &fe[s * steps..]);
                    let (sum, mag) = (0..rows / 16).fold((0.0f64, 0.0f64), |(y, m), sb| {
                        let isum: i32 = (16 * sb..16 * sb + 16)
                            .map(|v| code(w, v) * code(a, v))
                            .sum();
                        let t = half(we, sb) * half(ae, sb) * f64::from(isum);
                        (y + t, m + t.abs())
                    });
                    let g = f64::from(EXPERT_SCALES[place]) * f64::from(fs[s]);
                    let (want, band) = (
                        g * sum,
                        g.abs() * mag * 2.0f64.powi(-16) + f64::from(f32::from_bits(1)) / 2.0,
                    );
                    y_off += usize::from(if want.is_nan() {
                        !value.is_nan()
                    } else {
                        !value.is_finite() || (f64::from(value) - want).abs() > band
                    });
                }
                // `Edge`: every card column's Q8 block scale is f32::MAX / 127, its FP4
                // levels nonzero and its down outputs finite and nonzero.
                let edge_ok = values != Edge
                    || sel.iter().enumerate().all(|(s, &place)| {
                        place as usize >= n_card
                            || (d8[s * 2 * steps..(s + 1) * 2 * steps]
                                .iter()
                                .all(|&d| d == f32::MAX / 127.0)
                                && fq[s * rows / 8..(s + 1) * rows / 8]
                                    .iter()
                                    .all(|&w| (0..8).all(|n| (w >> (4 * n)) & 15 != 0))
                                && fused.0[s * hidden..(s + 1) * hidden]
                                    .iter()
                                    .all(|&v| v.is_finite() && v != 0.0))
                    });
                let replay_eq = bits_equal(&fused.0, &fused.1);
                let want_fault = if poison.is_some() {
                    Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn))
                } else if refused {
                    Some(Fault::at(LAYER_NONE, FaultSite::ExpertId))
                } else {
                    None
                };
                let fault_eq = plain.2 == want_fault && fused.2 == want_fault && plain.4 && fused.4;
                let pass = fp4_off == 0
                    && edge_ok
                    && y_kind_eq
                    && y_off == 0
                    && replay_eq
                    && fault_eq
                    && fused.3
                    && !plain.3
                    && x_fault == poison.map(|_| Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                println!(
                    "gemm32 case=nvfp4_gate_up rows={rows} k={k} slots={slots} varied={varied} values={values:?} scratch_saved_bytes={} refused={refused} poison={poison:?} fp4_off={fp4_off} edge_ok={edge_ok} y_kind_eq={y_kind_eq} y_off={y_off} eager_graph_eq={replay_eq} fault={:?} replay_fault_eq={} native={} {}",
                    2 * slots * rows * size_of::<f32>() + plain.7 - fused.7,
                    fused.2,
                    plain.4 && fused.4,
                    fused.3,
                    verdict(pass)
                );
                ok &= pass;
                if rows == 64 {
                    // The down's activations must hold K = the gate and up rows.
                    let mut g = DeviceBuffer::from_host(stream, &vec![SENT; slots * rows])?;
                    let mut u = DeviceBuffer::from_host(stream, &vec![SENT; slots * rows])?;
                    let mut wide = GemmAct32::new_gate_up_output(stream, slots, 2 * rows)?;
                    let refusal = dev.g32.enqueue_gate_up_sel32(
                        stream,
                        GateUp32Args {
                            w: [gate, up],
                            rows_per_expert: rows,
                            x: &x,
                            route: &route,
                            input: GemmInput::Shared { top_k },
                            sel: &sel_d,
                            n_card,
                            g: &mut g,
                            u: &mut u,
                            act: &mut wide,
                        },
                        sink,
                    );
                    let pass = refusal.is_err() && gpu.take_fault()?.is_none();
                    println!(
                        "gemm32 case=nvfp4_gate_up rows={rows} act_k={} refused_k={} {}",
                        2 * rows,
                        refusal.is_err(),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// Every 32-value case.
        pub(super) fn run(gpu: &Gpu, gk: &GemmKernels) -> Result<bool, GateError> {
            let dev = Dev {
                gpu,
                gk,
                g32: Gemm32Kernels::load(gpu.context())?,
            };
            let t0 = std::time::Instant::now();
            let mut ok = quant_case(&dev)?;
            ok &= swiglu_act_case(&dev)?;
            ok &= relu_sqr_case(&dev)?;
            ok &= dense_case(&dev)?;
            ok &= slab_case(&dev)?;
            ok &= routed_case(&dev)?;
            ok &= nvfp4_dense_case(&dev)?;
            ok &= nvfp4_routed_case(&dev)?;
            ok &= nvfp4_swiglu_case(&dev)?;
            ok &= swiglu_epi_case(&dev)?;
            ok &= nvfp4_native_case(&dev)?;
            ok &= nvfp4_gate_up_case(&dev)?;
            ok &= q5_0p_case(&dev)?;
            ok &= iq4nl_case(&dev)?;
            ok &= remap_case(&dev)?;
            ok &= f32tile_case(&dev)?;
            ok &= dense_mma_case(&dev)?;
            ok &= norm_gate_case(gpu, false)?;
            ok &= norm_form_case(&dev)?;
            ok &= bf16_store_case(&dev)?;
            ok &= mxfp8_w8a8_case(&dev)?;
            ok &= refusals(&dev)?;
            println!("gemm32 wall_s={:.1}", t0.elapsed().as_secs_f64());
            Ok(ok)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use bloomery_gpu_gates::exit_with;
            use std::process::ExitCode;

            fn assert_gate_failure<T>(result: Result<T, GateError>) {
                assert!(
                    result.is_err(),
                    "a partial reference is not a completed check"
                );
                assert_eq!(
                    exit_with("gate_gemm", result.map(|_| ())),
                    ExitCode::FAILURE
                );
            }

            #[test]
            fn checked_workers_must_complete() {
                let (rows, k, slots) = (2, 32, 3);
                let (w, decoded) = weights(false, rows, k, 73).expect("finite weights");
                assert!(decoded);
                let (a, _) = host_act(&activations(k, slots, 91), k, slots);
                let mut y = Vec::with_capacity(rows * slots);
                for s in 0..slots {
                    for r in 0..rows {
                        y.push(row_dot(&w, r, &a, s, false).0);
                    }
                }
                let run = |expert: &(dyn Fn(usize) -> Option<usize> + Sync),
                           col: &(dyn Fn(usize) -> usize + Sync),
                           y: &[f32],
                           slots| {
                    check(&w, false, rows, &a, expert, col, y, slots)
                };
                let clean = run(&|_| Some(0), &|s| s, &y, slots).expect("complete reference");
                assert_eq!(clean.checked, rows * slots);
                assert_eq!((clean.band_off, clean.bits_off, clean.host_bad), (0, 0, 0));
                let mut wrong = y.clone();
                wrong[0] = SENT;
                let mismatch = run(&|_| Some(0), &|s| s, &wrong, slots).expect("numeric mismatch");
                assert_eq!((mismatch.band_off, mismatch.bits_off), (1, 1));
                let host = run(&|_| None, &|s| s, &vec![SENT; y.len()], slots)
                    .expect("untouched host slots");
                assert_eq!((host.checked, host.host_bad), (0, 0));
                let written = run(&|_| None, &|s| s, &y, slots).expect("written host slots");
                assert_eq!(written.host_bad, slots);
                let empty = run(&|_| Some(0), &|s| s, &[], 0).expect("no work");
                assert_eq!(empty.checked, 0);

                let calls = AtomicUsize::new(0);
                assert_gate_failure(run(
                    &|_| Some(0),
                    &|s| {
                        if s == 1 {
                            assert_eq!(calls.fetch_add(1, Ordering::Relaxed), 0, "reference panic");
                        }
                        s
                    },
                    &y,
                    slots,
                ));
                assert_eq!(calls.load(Ordering::Relaxed), 2);
                assert_gate_failure(run(&|_| Some(0), &|_| panic!("reference panic"), &y, slots));
                assert_gate_failure(par_slots::<()>(
                    slots,
                    || panic!("reference initializer panic"),
                    |_, _| {},
                ));
            }

            #[test]
            fn gemv_comparison_workers_must_complete() {
                let (rows, k, slots) = (2, 32, 3);
                let (w, decoded) = weights_q50(rows, k, 74).expect("finite weights");
                assert!(decoded);
                let (a, _) = host_act(&activations(k, slots, 92), k, slots);
                let mut y = Vec::with_capacity(rows * slots);
                for s in 0..slots {
                    for r in 0..rows {
                        y.push(row_dot(&w, r, &a, s, false).0);
                    }
                }
                let clean = gemv_cmp(&w, rows, &a, &y, &y, slots).expect("complete comparison");
                assert_eq!(clean.checked, rows * slots);
                assert_eq!((clean.bits_off, clean.band_off, clean.nonfinite), (0, 0, 0));
                assert_gate_failure(gemv_cmp(&w, rows, &a, &y, &y[..y.len() - 1], slots));
                assert_gate_failure(gemv_cmp(&w, rows, &a, &y, &[], slots));
            }

            #[test]
            fn kquant_reference_workers_must_complete() {
                use super::super::{Input, Stack, check_ref, decode, dot_ref};

                let (rows, k, slots) = (2, 256, 3);
                let st = Stack {
                    name: "reference".into(),
                    ty: GemmWeight::Q4K,
                    bytes: synthetic(GemmWeight::Q4K, rows, 75).into(),
                    n_exp: 1,
                    rows,
                    k,
                    input: Input::PerSlot(1),
                };
                let ids = vec![0; slots];
                let codes = vec![vec![1; k]; slots];
                let d8 = vec![vec![0.5; k / 128]; slots];
                let rb = st.ty.block_bytes();
                let mut y = Vec::with_capacity(rows * slots);
                for s in 0..slots {
                    for r in 0..rows {
                        let dec = decode(st.ty, &st.bytes[r * rb..(r + 1) * rb], k);
                        y.push(dot_ref(&dec, &codes[s], &d8[s]).2);
                    }
                }
                let clean = check_ref(&st, &ids, &codes, &d8, &y).expect("complete reference");
                assert_eq!(clean.checked, rows * slots);
                assert!(clean.decode_ok && clean.fail.is_none() && clean.bits_differ == 0);
                assert_gate_failure(check_ref(&st, &ids, &codes, &d8, &y[..y.len() - 1]));
                assert_gate_failure(check_ref(&st, &ids, &codes, &d8, &[]));
            }

            /// A K-quant step whose product leaves the F32 normal range: `d8`
            /// 2^-149, inner sum `1.5 - 2^-24`, whose F32 rounding 1.5 would
            /// tie the subnormal product up to 2^-148; the exact inner sum
            /// gives 2^-149.
            #[test]
            fn kquant_contract_keeps_a_subnormal_step_exact() {
                use super::super::{Dec, dot_ref};

                let mut q = vec![0i8; 128];
                q[0] = -1;
                let dec = Dec {
                    grp: 32,
                    d: vec![2f64.powi(-24)],
                    dmin: vec![-1.5],
                    sc: vec![1; 4],
                    mn: vec![1; 4],
                    q,
                };
                let mut a = vec![0i8; 128];
                a[0] = 1;
                let contract = dot_ref(&dec, &a, &[f32::from_bits(1)]).2;
                assert_eq!(contract.to_bits(), 1);
            }

            /// The scale and norm-gate checks lean on [`nan_bits_equal`]: a
            /// shorter side is a difference, `-0.0` is not `0.0`, and any NaN
            /// matches any NaN.
            #[test]
            fn nan_bits_equal_takes_the_length_and_any_nan() {
                let nan = f32::from_bits(0x7fc0_0001);
                assert!(nan_bits_equal(&[1.0, -0.0, nan], &[1.0, -0.0, f32::NAN]));
                assert!(!nan_bits_equal(&[1.0, nan], &[1.0]));
                assert!(!nan_bits_equal(&[1.0], &[1.0, nan]));
                assert!(!nan_bits_equal(&[-0.0], &[0.0]));
            }
        }
    }
}
