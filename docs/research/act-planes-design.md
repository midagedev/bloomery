# Activation planes: one type for every model's quantized activations (round `actdesign`)

**Status.** Design memo from the read-only round `actdesign` (2026-09-27). 03 designs it and aa signs it before any
code (`docs/rebuild.md:341`). **aa signed §7 items 1–10 on 2026-09-27, with four conditions** (the end of §7). The
implementation rounds are §6 (`q3act` is steps B and C).

**What was read.**
- The tree at main `e5cc69a` (clean). Every line below is at that commit.
- q3tail's tables cite `ubatch.rs` at `ef00f78`. Their `:474/:527/:565/:589` are `:602/:655/:693/:717` here.
- The reference trees on the box:
  - ik_llama.cpp `c10fbbcc`
  - exllamav3 `0740edc`
  - mistral.rs `d5ae0f1`
  - the cuda-oxide pin `b9847e9`
  - cuda-core 0.3.1
- Headers read on the box (tensor-info tables only):
  - Qwen3-30B-A3B-Instruct-2507 Q4_K_M
  - DeepSeek-V2-Lite-Chat Q3_K_M
  - V4.1 Q3_K_M
  - Qwen3.6-35B-A3B UD-Q4_K_XL
  - GLM-5.3-Flash UD-Q2_K_XL
- GLM UD-Q4_K_XL's routed types come from the lead's disposition (`docs/research/modelspec-design.md:9-15`).

**How numbers are marked.** A number is either measured, with its source, or marked [derived].

## 0. On paper first

### 0.1 Decomposition

Every term below comes from the code or from a number already measured. No term needs a box run to settle the design. The only box run the design needs is step C's A/B (§6), and its card is in §6.

| Term | Source | Value |
|---|---|---|
| Bytes written per value, per plane | the layout in `crates/gpu/src/tensor.rs:203-226`; the stores in `lib.rs:622-721` | Q3, Q4 and Q6 are 1 B each. S8 is 0.125 B (one i32 per 32 values). D8 is 0.03125 B (one f32 per 128). All five together are 3.15625 B [derived] |
| What each writer stores today | every q8_1 writer stores all five planes (`lib.rs:806-814`: `requires` names all five) | code |
| What each reader reads | the launcher arguments (§2.1) | code |
| Quantize launches at U = 4096 | `docs/research/q3tail-design-report.md:115-119` (nsys, `ef00f78`, A6000) | measured: 94.76 µs for 60.0 MB, 185.45 µs for 120.1 MB, `gemm_swiglu_quant` 407.35 µs for 280.8 MB |
| Are those launches DRAM-bound? | same table: 634, 647 and 689 GB/s against a 690 GB/s floor | measured: 92–100 % of the floor |
| Is the card the wall? | `q3tail-design-report.md:79`: kernel sum 510.064 ms of a 511.127 ms window. `:94`: the host enqueue span is 5.404 ms | measured |
| The denominator today | pp4096 = 9,268 tok/s (`031b842`, rig-log 09-26#q3swz-pp), so 442.0 ms per prompt | measured, and the ms is [derived] |

**Prediction for Qwen3 (48 layers, U = 4096).**
- Unread writes per layer: 16.8 MB (attn_in) + 33.6 MB (attn_out) + 16.8 MB (ffn_in) + 50.3 MB (expert_down) = **117.4 MB** [derived].
- Per P 4096 prompt that is **5.637 GB** [derived].
- The time saved is **8.17–8.54 ms**, so the pp4096 ratio is **1.0188–1.0197** [derived] (§4.1).

### 0.2 Resource timeline

**(a) One Qwen3 attention-norm quantize launch** (`ubatch.rs:602`; U = 4096, K 2048, 65,536 one-warp blocks):

| Resource | Busy | Basis |
|---|---|---|
| Host CPU | about 5.6 µs to enqueue, well ahead of the card | 5.404 ms for 965 launches (`q3tail:79`, `:94`) [derived] |
| Card SM issue | 65,536 warps × I / (84 SMs × 4 schedulers × 1.5–1.6 GHz): **10–19 µs** for I = 80–150 warp-instructions [derived] | The body (`lib.rs:622-721`) has no loop over the block. Issue only binds past I ≈ 690 |
| Card DRAM | 33.6 MB read + 26.5 MB written = 60.0 MB. The floor is **87.0 µs**; measured **94.76 µs** | `q3tail:118` |
| PCIe | 0 | No quantized plane crosses PCIe. The hybrid handoff is f32 (`crates/gpu/src/hybrid.rs:14-16`) |
| Host DRAM, NVMe | 0 | The weights are resident on the card |

The launch's wall is the max of these, which is card DRAM. It runs on the card's one stream inside a card-bound window, so every µs removed from it comes off the wall.

**(b) Qwen3 prompt, per layer (U = 4096).**
- The card is busy about 9.2 ms [derived: 442.0 / 48].
- Of that, the three quantize launches plus `gemm_swiglu_quant` take 782 µs (8.5 %) [measured at `ef00f78`; those kernels are unchanged apart from `6123133`, see §4.1].
- The host spends about 0.11 ms enqueuing [derived], all of it in the card's shadow.
- PCIe carries the prompt image (49,152 B, q3rope) and the readback: µs.
- The wall is the card.

**(c) V4.1 prompt, per layer-batch (T = 512, lcg, no hot list).** All from `AGENTS.md`'s prefill paragraph:
- Host union: 66.7 ms (rig-log 09-26#uniondispatch-ab).
- Card route: 19.6 ms (#b1-pp-ab).
- The host's wait for the route: 0.8 ms (#prefillgroup-ab).

So the card works in the host's shadow. The activation writes this design removes are about 59.4 MB per layer-batch, which is about **86 µs of card DRAM [derived]**. Their effect on the wall is **≈ 0**.

**(d) Every decode path** (m ≤ 8, grids of a few blocks). The byte term sits below launch latency, so the effect is ≈ 0.

**Flow levers, checked before shortening any term:**
- **Asynchrony: 0.** The quantize's consumer (the GEMM) waits on it, and its producer (rms_norm) runs before it. A second stream is worth at most 2.0 ms, and it collides with F2 (`q3tail:271`, `:286`).
- **Bulk: already done.** There is one launch per site per ubatch.
- **SIMT: already done.** Each launch covers the whole batch.
- **SIMD:** the host tier is not touched by this design.
- **A flow change that removes a DRAM round trip.** Two exist:
  - F2, rms_norm and quantize in one pass: −7.7…−9.6 ms (`q3tail:242`).
  - F3, the quantize inside the flash epilogue: −5.7 ms (`q3tail:243`).

  Both call `q8_1_quant_vals` with the site's plane rows, so they come after this design and build on it.
- **Inline quantization in the consumer** (as in exllamav3's int8 gemv, `exl3_gemv_int8_kernel.cuh:19-45`): **rejected for the GEMM.** Every weight slab would re-read the f32 activation at 4 B/value instead of 1.16 B/value of planes, and would redo the quantization.

So narrowing the plane set is the right term to shorten. For Qwen3's prompt it comes straight off the wall, and it is what F2 and F3 need first.

## 1. The type

**Names.** `Act` is taken: modelspec already has `pub enum Act`, the activation function (`modelspec-design.md:285`). The proposal splits the job:
- `Plane` / `PlaneSet`: the vocabulary.
- `ActShape { planes, cap, k }`: the decision, which is the plan's sketch.
- `ActPlanes`: the card storage.
- Typed views for readers, and `ActWrite` for writers.

```rust
// crates/model — the format owner (modelspec §5); data only
pub enum Plane { F32, Q3, Q4, Q6, S8, D8, B32, HostQ8K, HostQ8x4 }
pub struct PlaneSet(u16);          // invariant (checked by `new`): any of Q3, Q4, Q6, S8 brings D8
impl PlaneSet {
    pub const fn of(p: &[Plane]) -> PlaneSet;                  // const: READS/WRITES are consts
    pub fn new(p: &[Plane]) -> Result<PlaneSet, ActError>;     // refuses a q8 plane without D8
    pub const fn union(self, o: PlaneSet) -> PlaneSet;
    pub const fn contains(self, p: Plane) -> bool;
    pub const fn within(self, o: PlaneSet) -> bool;
    pub const fn k_granule(self) -> usize;  // 256 with any q8 plane; 32 with B32 alone; 1 with F32 alone
}
pub enum Family { Dense, Experts, Gemm }   // dense gemv tiles <= 8 cols; expert gemv (sel/tiles); grouped int8 GEMM
pub fn card_reads(ty: GgmlType, f: Family) -> Option<PlaneSet>;   // None: no card kernel
pub fn host_reads(ty: GgmlType) -> Option<Plane>;                 // gguf `activation_format`, renamed

// crates/gpu/src/act.rs — card storage (03)
pub struct ActShape { planes: PlaneSet, cap: usize, k: usize }
impl ActShape {
    pub fn new(planes: PlaneSet, cap: usize, k: usize) -> Result<ActShape, GpuError>;
    // refused by name: empty set, cap 0, k not a multiple of k_granule, a q8 plane with k > ACT_MAX_K (20,480)
    pub fn bytes(&self) -> usize;           // the one allocation formula
}
pub struct ActPlanes {                      // an absent plane is a zero-length DeviceBuffer
    shape: ActShape,
    f32: DeviceBuffer<f32>, q3: DeviceBuffer<u64>, q4: DeviceBuffer<u32>, q6: DeviceBuffer<u32>,
    s8: DeviceBuffer<i32>, d8: DeviceBuffer<f32>, b32: B32Planes,   // B32: Q8Blocks32's codes, sums, scales
}
impl ActPlanes {
    pub fn new(stream: &CudaStream, shape: ActShape) -> Result<ActPlanes, GpuError>;      // load time only
    pub fn view<'a, V: ActView<'a>>(&'a self, cols: Range<usize>) -> Result<V, GpuError>; // V::READS ⊆ planes, cols.end <= cap
    pub fn write(&mut self, set: PlaneSet, cols: Range<usize>) -> Result<ActWrite<'_>, GpuError>;
    pub fn read_back(&self, s: &CudaStream, cols: Range<usize>) -> Result<ActHost, GpuError>; // gates and taps
}
pub trait ActView<'a> { const READS: PlaneSet; }
// Q3View{q3,d8,cols,n_sb}  Q4View{q4,s8,d8,..}  Q6View{q6,d8,..}  GemmView{q6,s8,d8,..}  B32View  F32View
// ActWrite: each plane's window and its row count (`cols` when the plane is in the set, else 0)
```

**Absent planes cost nothing, and the kernel ABI does not change.** `DeviceBuffer::zeroed(stream, 0)` allocates nothing and stores a null pointer (`cuda-core-0.3.1/src/simt/device_buffer.rs:549-555`). The launch packet then passes an aligned sentinel for a zero-length slice (`cuda-host/src/launch.rs:149-177`), for read-only arguments (`:191-199`) and writable ones (`:215`). So every kernel keeps all of its plane parameters, and an absent plane arrives as `(sentinel, 0)`.

**Views are host address arithmetic.** A column window is `tensor::window` (`tensor.rs:21-30`) at `c0 · column_length` into each plane. The quantize entry already takes an input offset `x0` for the same purpose (`lib.rs:783-785`).

**One cap, not one per plane.** Today each arena serves a single family:
- the decode `Arena`: m ≤ 8
- `UbArena`: U, or slots
- the V4.1 batch arenas: T

So one cap per shape is enough. A cap per plane only pays if the decode `Arena` and `UbArena` merge. In that case Q4 held at the ubatch cap would cost **58.7 MB** (50.3 MB without padding) [derived]. That merge is aa's call under `step_rows`, so per-plane caps are deferred.

**One type for card and host?** One vocabulary and one table, but two storage types. The reasons:
1. The memory differs: device versus pinned host.
2. The writers differ: card kernels versus AVX2 `qdot::quantize_col` (`crates/qdot/src/lib.rs:230-273`).
3. The layouts differ: the q8_1 permutations versus `block_q8_K` / `q8_2_x4`.
4. The lifetimes differ: a load-time arena with capture-stable addresses versus per-worker buffers.
5. No quantized plane crosses PCIe (the handoff is f32, `hybrid.rs:14-16`), so no single buffer ever needs both forms.

**What it replaces.**
- `Q8Act`: `tensor.rs:191-320`
- `GemmAct`: `gemm.rs:1803-1905`
- `MxAct`: `gpu-deepseek41/src/experts_mxfp4.rs:731-786`
- `Q8Blocks32`: `q5.rs:838-918`; its three buffers become the `B32` plane
- `Q8ActHost`: `fused.rs:894-910`; it becomes `read_back`

58 files name one of these five types; 54 of them name `Q8Act` (grep at `e5cc69a`).

**B32 stays a `Plane`,** so that the format owner's Q5_0/Q5_1 row has one answer. `v2fence` moves its writer and readers under `arch/deepseek2` (`rebuild.md:80`, `:350`). When V2-Lite retires, the variant is deleted, which is a deletion-class change.

**Arena memory, from the padded formula (`tensor.rs:277-281`).** Qwen3 `UbArena` holds `act_hid` K 2048 × 4096, `act_attn` K 4096 × 4096 and `act_h` K 768 × 32,768 (`ubatch.rs:157`, `:162`, `:170`). It is **184.0 MB today and 66.6 MB under the design** [derived]. At K 768 and K 512 the padded columns are 1.32× the bytes actually written.

## 2. Who decides the plane set, and when

The set is decided **at load, after placement**. It comes from each consumer's GgmlType (from the header), the family of the path that runs it, and its tier. It is looked up in one table, which lives with the format owner (`modelspec-design.md:685-697` gains the columns below). It is not written into `arch/` code.

### 2.1 The table: card reads

| Type | Dense (gemv) | Experts (gemv) | Gemm | Host |
|---|---|---|---|---|
| Q3_K | Q3·D8 (`lib.rs:2640-2720`; `model/kernels.rs:509-512`) | Q3·D8 (`moe_fused.rs:318-319`) | Q6·S8·D8 (`gemm.rs:600-650`, `:2388-2390`) | HostQ8K |
| Q4_K | Q4·S8·D8 (`lib.rs:2566-2621`; `arch/qwen3moe/proj.rs:318-320`) | Q4·S8·D8 (`q4k_sel.rs:778-780`; `arch/qwen3moe/experts.rs:389-391`) | Q6·S8·D8 | HostQ8x4 |
| Q5_K | F32 (the dense f32 gemv, `gpu-deepseek41/src/dense.rs:1-40`) | — (placement drops it: `of_routed`, `placement.rs:366-368`) | Q6·S8·D8 (`gemm_q5k`, `gemm.rs:138-176`) | HostQ8x4 |
| Q6_K | Q6·D8 (`lib.rs:2936-2977`; `head_argmax.rs:395-396`) | Q6·D8 (`q6k_sel.rs:320-321`) | Q6·S8·D8; Q6·D8 after step E | HostQ8x4 |
| Q5_0, Q5_1 | B32 (`q5.rs:1231`) | B32 (`q5.rs:1326`, `:1413`) | — | HostQ8x4 |
| Q8_0, F32, BF16 | F32 (`q8f32.rs:1-7`: those kernels take f32 activations) | F32 | — | — / plain |
| MXFP4 | — | Q4·D8 (draft, `experts_mxfp4.rs:731-786`) | — | HostQ8x4 |
| IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K | — | Q4·D8 (`iq.rs:56-58`, reads at `:419-453`). Q2_K builds its 16-value sums from the codes inside the kernel (`iq.rs:66-67`). Placement refuses them | — | IQ3_XXS: HostQ8K; the others: — |

**Why the GEMM column carries S8 even where it isn't used.** Today the GEMM stages S8 for every entry (`gemm.rs:636-648`), including `gemm_q6k` and `gemm_q3k`, which never read it. Step E narrows those rows to Q6·D8. ik's per-type layouts are the precedent: Q3_K and Q6_K take D4, and Q4_K and Q5_K take DS4 (`mmq.cuh:55-110`).

This table generalizes the existing `reads_q8_1` predicates. They are defined three times (`dense.rs:770-772`, `chain/attn.rs:2384-2389`, `:2434-2441`) and named 21 times.

### 2.2 The functions

```rust
pub struct Consumer { tensor: String, layer: usize, ty: GgmlType, family: Family, tier: Tier }
/// Union of the consumers' reads. A card consumer with no kernel becomes an Unimplemented item;
/// a host consumer adds F32 (the handoff) and gets host_reads(ty) on its own side.
pub fn site_planes(c: &[Consumer]) -> Result<PlaneSet, Vec<Unimplemented>>;
/// At load, after placement: every (layer, site) set, the union over layers per site, k, and cap
/// (from `step_rows`). Every missing kernel is refused at once:
/// PlacementError::Unimplemented (placement.rs:78-84, :764-767).
pub fn act_plan(spec: &ModelSpec, t: &ModelTensors, placed: &Placement, rows: &StepRows)
    -> Result<ActPlan, PlacementError>;
```

**Sets can differ per layer, and that is free.** The arena allocates the union across layers. Each launch passes its own layer's rows, which are baked into the captured node's arguments. Qwen3 needs this: v is Q6_K on only 24 of its 48 layers.

**Before `modelspec` lands,** each arena calls `site_planes` itself, with the consumer types read from the header.

### 2.3 Sets per model

Notation: gemv is the decode/pass path (cap 1 or 8). Gemm is the prompt path.

**Qwen3-30B-A3B Q4_K_M** (48 layers)

| Site (writer) | K | Consumers | gemv | Gemm (cap U = 4096 / 32,768 slots) |
|---|---|---|---|---|
| attn_in (`norm_quant`; `ubatch.rs:602`) | 2048 | q, k: Q4_K. v: Q4_K on 24 layers, Q6_K on 24 (0-5, 8, 11, …, 38, 41-47) | Q4·S8·D8, plus Q6 on the Q6_K-v layers | Q6·S8·D8 |
| attn_out (`dispatch.rs:337`; `ubatch.rs:655`) | 4096 | o: Q4_K | Q4·S8·D8 | Q6·S8·D8 |
| ffn_in (`router.rs:639` at m = 1; `ubatch.rs:693`) | 2048 | router (F32); expert gate and up (Q4_K) | F32·Q4·S8·D8 | F32 (the router, `ubatch.rs:697`)·Q6·S8·D8 |
| expert_down (`dispatch.rs:411`; `ubatch.rs:717`) | 768 × 8 | Q6_K on 24 layers, Q4_K on 24 | Q6·D8 / Q4·S8·D8 | Q6·S8·D8 |
| head | 2048 | output: Q6_K | Q6·D8 | — |

**V4.1 Q3_K_M** (40 layers). The same sets on both paths.
- attn_in (K 5120; q_a, kv, compressor, indexer proj): Q3·D8
- q_latent (K 1280): Q3·D8
- kv_pre (K 512, layers 2, 8, 14, 20): Q3·D8
- heads (8 × K 4096): Q3·D8
- wo_a (K 8192): Q3·D8
- engram (K 6144): Q3·D8
- ffn_in (K 5120; router BF16, Q3_K card and shared experts, host handoff): F32·Q3·D8
- down (K 2304): Q4·S8·D8 on layers 2-39; F32 on layers 0-1 (Q5_K, dense f32)
- hc (K 20480): F32 only (`hc_pre` quantizes in registers, `hc.rs:14-22`)
- head (K 5120): Q6·D8

**V2-Lite Q3_K_M** (27 layers, decode)
- attn_in (K 2048): Q3·D8
- kv_b (K 512): Q3·D8
- attn_out (K 2048, Q4_K): Q4·S8·D8
- ffn_in (K 2048; router F32, Q3_K dense, expert and shared gate/up): F32·Q3·D8
- dense down (layer 0, K 10944, Q5_1): B32
- expert down (K 1408 × 6, Q5_0): B32
- shared down (K 2816, Q4_K, the pair's arm b, `q5.rs:429-524`): Q4·S8·D8
- head: Q6·D8

**Qwen3.6-35B-A3B UD-Q4_K_XL** (30 GDN layers; 10 GQA layers at 3, 7, …, 39)
- Every dense projection is Q8_0 (the GDN qkv/gate/ssm_out, the GQA q/k/v/o, the shared expert, the output), plus ssm_alpha/beta F32. Their gemv set is F32. The GEMM family is **refused**: there is no Q8_0 GEMM.
- ffn_in (K 2048): F32·Q4·S8·D8. A Q5_K gate/up on layer 1 placed on the card is **refused** (no Q5_K expert gemv).
- expert_down (K 512 × 8): Q5_K on 36 layers is **refused** on the card. Q6_K on layers 1, 34, 38, 39 takes Q6·D8.
- The Gemm set is Q6·S8·D8. That needs `gemm_q5k`.

**GLM-5.3-Flash UD-Q2_K_XL** (34 KDA and 11 MLA layers, plus nextn 45)

| Site | gemv | Gemm |
|---|---|---|
| KDA in (K 4096: q Q5_K; k, v Q6_K; f_a/g_a/beta Q8_0) | F32·Q6·D8 | Q6·S8·D8; the Q8_0 GEMM is **refused** |
| f_b/g_b (K 128, Q8_0) | F32 | refused (Q8_0) |
| KDA o (K 8192, Q5_K) | F32 | Q6·S8·D8 |
| MLA in (K 4096: q_a Q5_K, Q6_K on 11; kv_a and indexer Q8_0) | F32, plus Q6·D8 on 11 | Q6·S8·D8 |
| MLA q_b (K 1536, Q8_0) | F32 | refused (Q8_0) |
| MLA k_b, v_b (Q8_0) | F32 | refused (Q8_0) |
| MLA o (K 16384, Q5_K; Q6_K on 11) | F32 | Q6·S8·D8 |
| dense layers 0-2: gate/up Q5_K | F32 | Q6·S8·D8 |
| dense layers 0-2: down Q6_K (K 12288) | Q6·D8 | Q6·S8·D8 |
| routed IQ2_XS/IQ3_XXS/IQ4_XS stacks | host tier (F32 handoff; placement refuses them on the card) | — |
| shared down (Q6_K; Q8_0 on 11) | Q6·D8 | Q6·S8·D8 |
| hc (Q8_0, K 16384) | F32 | — |
| output (Q4_K) | Q4·S8·D8 | — |

For **GLM UD-Q4_K_XL**, the routed gate/up are Q4_K (layer 11 is Q5_K), and the routed down is Q5_K on 40 layers and Q6_K on 11, 12 and 44. The card Q5_K experts are refused (the host tier takes them), and the GEMM needs `gemm_q5k`.

## 3. Writers and readers

**Writers.**
- `q8_1_quant_vals` (`lib.rs:622-721`) is the only owner of the body and of the refusal. It gains one row count per plane: `q3_rows`, `q4_rows`, `q6_rows`, `s8_rows`. Each is `u32` and is either 0 or `m_cols`.
- A plane is stored only when `col < rows_p`. That condition is warp-uniform, because one warp quantizes one column's 128-value block. When Q3 is absent, the Q3 pair shuffle (`shuffle_xor(8)`) is skipped; when S8 is absent, the S8 butterfly is skipped.
- The contract becomes `q3.len() >= q3_rows * 64 * half_it`, `q4.len() >= q4_rows * 256 * quad_it`, `q6.len() >= q6_rows * 128 * half_it`, `s8.len() >= s8_rows * 8 * n_sb`, and `d8.len() >= m_cols * 2 * n_sb` (replacing the current contract at `lib.rs:808-814`), plus `m_cols >= q3_rows`, and so on.
- The `requires` grammar at pin `b9847e9` allows all of this:
  - arithmetic may nest (`contract.rs:461`, `:595`);
  - `.len()` is the only method allowed (`:540-545`);
  - signed integer parameters are refused (`:523`), which is why the rows are `u32`;
  - there is no `||`, so "rows ∈ {0, m}" is enforced by the host launcher.
- **The ten entries that change:**
  1. `q3k_quantize_q8_1` (`lib.rs:781-865`)
  2. `norm_quant` (`fused.rs:115`)
  3. `gemm_swiglu_quant` (`gemm.rs:1712-1800`)
  4. `q8_1_quantize_ord` (`q4k_sel.rs:513`)
  5. `q8_1_quantize_sel` (`q4k_sel.rs:591`)
  6. `flash_latent_q8` (`flash.rs:941`)
  7. `flash_merge_q8` (`flash.rs:1294`; both flash entries go through `quant_row`, `:1441`)
  8. `qwen3moe_router_norm` (`router.rs:639`, calling `:756`)
  9. arm b of `q5_q8_1_quantize_pair` (`q5.rs:429`)
  10. `dflash_quantize_q8_1` (`experts_mxfp4.rs:166`)
- **F32 is a plane too.** `norm_quant` stores `y` only when F32 is in the set. Today it stores `y` always and says so: "a site with no f32 consumer passes a buffer it is about to overwrite anyway" (`fused.rs:82-85`).
- Each writer declares `WRITES: PlaneSet`. `act_plan` checks at load that each site's set is within its producer's `WRITES`, so asking a writer for a plane it cannot produce is a named error at load. `ActPlanes::write` checks the same thing again.

**Readers.**
- Each launcher takes a typed view (`Q3View`, `Q4View`, `Q6View`, `GemmView`, `B32View`, `F32View`) whose `READS` constant is fixed.
- A view can only be built through `view::<V>()`, which refuses by name any set missing a plane that `V::READS` needs.
- A unit test in `crates/gpu` checks that every row of the table in §2.1 equals the `READS` of its kernel's view, so the table cannot drift from the kernels.

**The column count is a launch argument.** Today launchers read the count back from the buffer with `act.m()`:
- `lib.rs:2498`, `:2572`, `:2647`, `:2754`
- `fused.rs:568`
- `proj.rs:244`, `:305`, `:373`
- `experts.rs:290`
- `q5.rs:1041`, `1095`, `1106`, `1196`, `1290`, `1378`
- `q4k_sel.rs:711`, `890`, `960`, `1014`
- `q6k_sel.rs:256`, `moe_fused.rs:234`, `head_argmax.rs:359`, `router.rs:1318`

Under the design the count is the view's `cols`, which is the caller's own count. The shape's cap is only an allocation bound, checked when the view is made.

The 8-column limit becomes a check in the launcher on `view.cols`. `ColGroups` (`lib.rs:283-340`) is the precedent. This removes DS2's API cap (`rebuild.md:192`).

**NaN and infinity: unchanged.**
- `q8_1_quant_vals` ballots `quad_finite`.
- A refused block gets a NaN D8, zero codes and a zero sum, and the function returns `refused`.
- The entry then raises QuantColumn (NormQuant for a norm scale, Q5Quant for B32, HcQuant in `hc_pre`) into the fault word, `(layer<<8)|site` by atomic min.

The only difference is that the zero codes of a refused block land only in the planes that are present. D8 is required in every q8 set, so the NaN still reaches every q8 dot. No reader ever reads an absent plane.

**Alternatives considered and rejected.**
- A const-generic plane mask on the kernel. It multiplies the entries, and a const-generic `#[kernel]` hits E0401 on static `SharedArray` and hashes the entry names.
- A quantize entry only for `GemmAct` (q3tail's Q). That is a second path.
- An `Option<DeviceBuffer>` per plane. Zero-length buffers already carry absence through the ABI, so the Option only adds unwraps.

## 4. Bytes and time [derived]

### 4.1 Qwen3, prompt path

| Site | Bytes per token, today → design | MB per layer at U 4096 |
|---|---|---|
| attn_in (K 2048) | 6464 → 2368 | 26.5 → 9.7 |
| attn_out (K 4096) | 12928 → 4736 | 53.0 → 19.4 |
| ffn_in (K 2048) | 6464 → 2368 | 26.5 → 9.7 |
| expert_down (8 × K 768) | 19392 → 7104 | 79.4 → 29.1 |
| **per layer** | 45,248 → 16,576 | 185.3 → 67.9 (**−117.4 MB**) |
| **per P 4096 prompt** | | 8.896 → 3.259 GB (**−5.637 GB**) |

Two models for how launch time follows bytes:
- **(a) constant efficiency:** time is proportional to bytes at the launch's own GB/s.
- **(b) constant overhead:** the measured time minus the floor stays fixed.

| Launch (line at HEAD) | Per layer | MB, today → design | Measured µs | Saved µs, (a) / (b) |
|---|---|---|---|---|
| quantize, attention norm (`:602`) and FFN norm (`:693`) | 2 | 60.0 → 43.2 | 94.76 | 26.5 / 24.3 each |
| quantize, attention output (`:655`) | 1 | 120.1 → 86.5 | 185.45 | 51.8 / 48.6 |
| `gemm_swiglu_quant` (`:717`) | 1 | 280.8 → 230.5 | 407.35 | 73.0 / 72.9 |
| **per layer** | | | 782.3 | **177.8 / 170.1** |

- Per prompt that is **−8.17…−8.54 ms**. At 442.0 ms the ratio is **1.0188–1.0197**, so pp4096 goes from 9,268 to 9,442–9,450.
- At P 512 the band is +1.2…+2.2 %. The planes fit in the 6 MB L2 there, so the byte model is weak.

**Reconciliation with the audit.** The audit says "about 5.6 GB, 8–9 ms ≈ 1.5–1.7 %" (`docs/research/audit/qwen3.md:218`) and q3tail's Q row says −8.2…−8.5 ms (`q3tail:278`).
- The bytes agree: 5.637 GB.
- The milliseconds agree: 8.17–8.54 ms.
- The percentage rose because the window shrank from 511 ms to 442.0 ms: q3gemmb and q3swz landed in between. The denominator moved, not the term.

**Caveat.** The measured times come from `ef00f78`, before `6123133` added the refusal ballot. That adds a few instructions to a DRAM-bound kernel, so its effect should be ≈ 0. Both A/B arms carry it anyway.

**The S8 plane on the Q6_K-down layers** (+75.5 MB per prompt) stays until step E. Dropping the write before the GEMM stops staging it would leave the GEMM reading bytes nobody wrote.

### 4.2 V4.1

| Site | Bytes per token, today → design |
|---|---|
| attn_in | 16160 → 5280 |
| q_latent | 4040 → 1320 |
| heads | 103424 → 33792 |
| wo_a | 25856 → 8448 |
| ffn_in | 16160 → 5280 |
| down (38 layers) | 7272 → 2664 |
| kv_pre (4 layers) | 1616 → 528 |

- In total that is 6.91 → 2.27 MB per token (−67 %), plus **4,608 B per card expert slot**.
- With CED off, a P 4096 prompt writes **19.0 GB less**.
- With CED on, the plan runs 106,400 of 163,840 layer-positions in full plus 4,088 latent-only (`tools/flow/plans/ds41-p4096-ced-on.rec`, the `call need` line; `body/ced.rs:118-126`). That makes it about **12.4 GB less** [derived, assuming every layer saves the same].
- The DRAM floor is 18–28 ms, but that time sits under the host's shadow (§0.2 c), so the wall effect is **≈ 0**.
- The value is structural: it is DS2's precondition. That is why V4.1 files no A/B for this change: its effect is inside the ruler and in the shadow.

### 4.3 V2-Lite, decode

It writes 754,672 → 262,640 B per token, which is 0.71 µs per step at 690 GB/s. Decode is latency-bound, so the effect is ≈ 0.

### 4.4 Qwen3.6 and GLM

These are counterfactuals: what the prompt GEMM sites would write if they were wired with all five planes, as Qwen3's are today.
- **Qwen3.6:** 775,680 → 284,160 B per token, **−2.013 GB** per P 4096 (a floor of 2.9 ms).
- **GLM Q2_K_XL:** 2,999,296 → 1,098,752 B per token, **−7.785 GB** (11.3 ms).

## 5. References

| | ik_llama.cpp | mistral.rs | exllamav3 | this memo |
|---|---|---|---|---|
| Form of the int8 activation | gemv: 32-value q8_1 with a half2 of (d, sum) (`quantize.cu:13-47`). MMQ: `block_q8_1_mmq`, 128 codes plus 16 B in three layouts (`mmq.cuh:26-51`) | the same two forms (`fast_mmvq.rs:19-20`, `fast_mmq.rs:80-100`) | fp16 plus a per-weight input transform (`exl3_gemm.cu:23-34`). int8 only inline, in the mul1 gemv (`exl3_gemv_int8_kernel.cuh:19-45`; not graph-capturable, `exl3_gemm.cu:177-186`) | one scale per 128 values, sums per 32, three code permutations |
| Who picks the layout, and when | the consumer's weight type, per graph node (`mmq.cuh:55-110`, `ggml-cuda.cu:2564`) | the consumer's weight type, per call | per weight: one temporary per consumer weight (`exl3_moe_common.cuh:26-30`, `temp_state_g/_u`) | the union over the site's consumers, at load |
| Two consumers needing different layouts | quantize again. The QKV fusion shares one quantization only when the types match (`ggml-cuda.cu:2527-2547`) | `shared_lhs` requires the same dtype (`fast_mmq.rs:528-540`) | one temporary each | one launch writes the union |
| Buffer | pool allocation per node (`ggml-cuda.cu:2520`) | a leaked workspace that only grows, sized to the next power of two, with stable addresses for graph replay (`fast_mmvq.rs:88`, `:101`; `fast_mmq.rs:346`, `:604-613`) | temporaries | a load-time arena (capture-stable; no allocation in the step) |
| gemv/GEMM cut | `ne[1] <= 8` (`ggml-cuda.cu:3602-3631`) | `MMVQ_MAX_BATCH = 8` (`fast_mmvq.rs:53`) | 144 rows, then weights are reconstructed (`exl3.py:10`, `:133-139`) | the family of the path |
| Expert-ordered quantization | `quantize_mmq_q8_1_id` (`quantize.cu:179-`) | `quantize_mmq_q8_1_glu(…, ids, …)` (`mmq_quantize.cu:238`) | — | not yet (see the improvement list) |
| The sum | a sum of f32 values (`quantize.cu:123`) | same as ik | — | the integer sum of the codes; kept, because it is part of our bit contract |

**What the memo follows.** It keeps ik's and mistral.rs's rule: the writer writes exactly what its consumers' weight types read. It departs from them in two places:
1. **The unit of decision is a site, decided at load.** One launch serves a union of consumers. For example, V4.1's ffn_in feeds its experts and its shared expert at once, and Qwen3's attn_in on a Q6_K-v layer feeds both Q4_K and Q6_K.
2. **The storage is a load-time arena.** mistral.rs grows a leaked workspace for the same address stability; we size ours at load instead.

## 6. Migration

| Step | Class | Proof | Timed A/B |
|---|---|---|---|
| **A** `actplan`: `Plane`, `PlaneSet`, `Family`, `card_reads`, `host_reads`, `site_planes`, in `crates/model/src/act_reads.rs` (seed of the format owner) | host-only; adds a new file | `just check`, `just lint` (≤ 133), a unit test that `host_reads` equals `activation_format` for every type | none |
| **B** `q3act-1`: `ActPlanes` everywhere; every plane still allocated and written; counts become launch arguments; per-m `Vec`s become one arena per site | move | `just ptx-scan` identical for `generate_qwen3moe` (`--features gpu`), `gate_gemm`, `gate_p8`, `gate_qwen3moe_e2e`, `generate_ds41` (`--features gpu,deepseek41`) and `gate_e2e`. Structural lines unchanged: 604 / 601 / 673 nodes, eager = replay, the six ubatch `--dump` md5s, and V4.1's `--plan` launch list | none |
| **C** `q3act-2`: row counts in the ten writers (plus `norm_quant`'s F32); Qwen3's arenas take their table sets; V4.1 and V2-Lite keep every plane (their sets change in D and at retirement) | moves PTX: exactly the ten writer rows; `gate-ptx-spill` re-pin | `gate-gpu-gemm`: `y_fnv` 328/328, with the plane comparisons at `gate_gemm.rs:1105-1116` and `:1651-1659` narrowed to the set. **New FAIL-first clause:** absent planes, allocated and filled with a sentinel, stay untouched (a writer that ignores its zero row count turns the clause red). Also: `gate-gpu-qwen3moe-e2e` (both GQA arms), the md5s, the V4.1 gates (step, prefill, long, faults), V2-Lite (e2e, hybrid), and the fault cases in gate_gemm, gate_hybrid, gate_p1, gate_p5, gate_q4k_sel, gate_qwen3moe_router and gate_deepseek41_step | **one, 4 rounds.** Judged on pp4096; P 512 recorded, not judged |
| **C′**: the gemm items gemmsplit left (the clamp `gemm.rs:1186-1187` becomes a contract plus a fault; the `requires` gaps near `:620-624`; R11 at `:1356-1357`) | moves PTX; its own commit, so each byte change has one reason | its own `ptx-scan` diff plus `gate-gpu-gemm` | none |
| **D** (aa, with `batchwide`): V4.1 gets one `ActPlanes` per site at cap T; the `dtod` copies become window writes; `MxAct` is folded in; `reads_q8_1` goes to the table | host; launch count −≈128 per layer-batch [derived] | V4.1 bit gates and the `--plan` launch count | inside `batchwide`'s own prose A/B |
| **E**: Q3_K/Q6_K GEMM entries stop staging S8 (`gemm.rs:636-648`) | moves PTX | `gate-gpu-gemm`, the md5s | none: −75.5 MB ≈ −0.11 ms (0.025 %), inside the ruler |
| **F**: F2 and F3 | moves PTX | each its own gates | one each |

**Order and file boundaries.**
- gemmsplit, then B, then C, then C′, E and F.
- A can land any time before C.
- D goes after C and after aa's `modelspec`.
- B's files overlap aa's work in two places:
  - `gpumodel`: `crates/gpu/src/model/{kernels,probe}.rs`.
  - `v2fence`: `arch/deepseek2/*`, `q5.rs`, `flash.rs`.

  So B goes after `gpumodel` lands, and before or after `v2fence` runs — not while it runs. Before is better, because `v2fence` then moves code that already has its new names.
- **B's recommendation: no shim.** B makes aa's 11 files (and the V4.1 gates) mechanical changes at an aa wave boundary, proven by `ptx-scan` identity, and aa signs that diff. If there is no such window, the fallback is a `Q8Act` shim that D deletes.

**Step C's card** (`docs/cards/q3act-pp.card`; the ruler is h = 1.04 % at 4 rounds with sd 0.6, and the band's near edge of 1.5 clears it):
```
kind: ab
question: does writing only Q6·S8·D8 shorten the Qwen3 prompt at P = 4096 by the bytes the quantizers stop writing?
term: pp tok/s @ P = 4096, n = 0, A6000, Qwen3-30B-A3B Q4_K_M - (q3act - base) / base, paired by round
unit: %
predict: +1.5..+2.4
rounds: 4
condition: the q3act-2 tree and its base (a bin: arm) alternate in one lease, BLOOMERY_AB_ROUNDS=4 just
  depth-gpu-qwen3moe 512 bin:<base>:512 4096 bin:<base>:4096, order rotated every round; no row carries
  [other-busy]; P = 512 and the decode rows are screens
decide-in: the per-launch byte model holds; price F2 with the same bytes and open it
decide-below: the launches lost GB/s at fewer bytes: an nsys of both arms, the four launches against model (b), before F2
decide-above: the GEMM's reads got faster too (L2): an nsys of both arms, re-fit the GEMM term before F2
```

## 7. What aa signs, and one lead decision

1. `Q8Act` becomes `ActPlanes` plus typed views, including in `gpu-deepseek41` (11 files) and the V4.1 gates. The 1..=8 cap in `with_k` leaves the type; it becomes a launcher check on the view.
2. `MxAct` (`experts_mxfp4.rs:731-786`) becomes an `ActPlanes` of Q4·D8.
3. `dflash_quantize_q8_1` (`experts_mxfp4.rs:166`) takes the plane row counts. That moves PTX in `generate_ds41` in step C, but V4.1's bits do not move.
4. V4.1's per-count `Vec`s (`chain/attn/batch.rs:298-327`, `chain/ffn/batch.rs:927-941`) become one arena per site. The two `dtod` copies per chunk (`chain/ffn/batch.rs:1925-1944`) become window writes (step D).
5. `reads_q8_1` (3 definitions, 21 mentions) is replaced by `card_reads(ty, Family)`.
6. The format owner (`modelspec-design.md:685-697`) gains the columns Dense, Experts and Gemm. `activation_format` becomes its host column.
7. `act_plan` runs at load after placement. A missing kernel is a `PlacementError::Unimplemented` item, listed together with all the others.
8. B32 stays a plane. `v2fence` moves its writers and readers, and V2-Lite's retirement deletes it.
9. B edits aa's files mechanically at a wave boundary (fallback: a shim).
10. The host tier keeps its own storage (`quantize_col`), and the handoff stays f32.

**aa's conditions (2026-09-27).**
- On 1: `with_k`'s refusal of a count outside 1..=8 does not disappear. It moves to the view's launcher check and stays
  a named error (no silent truncation), and every V4.1 launcher that takes m states the m it accepts the way
  `gpumodel`'s `Rows` (m ≤ 8) does.
- On 3: step C's proof adds the DSpark gates (`gate-gpu-dspark-graph`, `-kv`, `-hc`, `-experts`, and
  `ds41-dspark-loop`): `dflash_quantize_q8_1` is the draft's quantizer. Its `ptx-scan` diff shows that only the
  writer rows moved.
- On 9: B lands after `gpumodel` and not while `v2fence` runs, rebased onto aa's `del2` (whose D1 deletes engine APIs
  only `gpu-spike` called, possibly `Q8Act::new`). Before B lands, 03 sends aa the list of hunks in aa's files
  (`gpu-deepseek41`'s 11 and the V4.1 gates) for review. The proof is §6's (`ptx-scan` identical for `generate_ds41`
  and `gate_e2e`, the V4.1 `--plan` launch list, the node counts).
- On 6: aa's `modelspec` spec takes the three columns (Dense, Experts, Gemm) and the host column. If step A lands
  before `modelspec`, `modelspec` moves `act_reads.rs` inside the format owner, so there is one table.
- Also: item 5 can close aa's triage item on `body/prefill.rs`'s `reads_q8` and `shadow_entries`, which look a weight
  up by its name as a `String` every layer-batch: `card_reads` resolved at open into per-layer constants.

**Lead decision** (decided by the user on 2026-09-27: `gemm_q5k` stays). Deleting `gemm_q5k` is on hold "until the IMMA decision" (`docs/rebuild.md:302`). Deleting it would empty the (Q5_K, Gemm) cell of the table. Qwen3.6 routes down through Q5_K on 36 of its 40 layers, and GLM UD-Q4_K_XL on 40 of its 43 (`modelspec-design.md:9-15`). Without `gemm_q5k`, neither next model has a down GEMM on its prompt path.
