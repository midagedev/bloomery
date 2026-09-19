# mistral.rs prior art — what it does, and what bloomery should take from it

Date: 2026-09-19. Upstream: `EricLBuehler/mistral.rs`, branch `master` at
`d5ae0f18` (last push 2026-09-08, i.e. the tree has been still for 11 days).
Crate version 0.9.3. All line numbers below are against that revision unless
noted; mistral.rs pins `huggingface/candle` at rev `35d7ae7` (root
`Cargo.toml`), and candle line numbers are against that rev.

Method: read the code via `gh` + raw fetches only. Nothing cloned, nothing
built, nothing run. Every code claim carries a path and a line; the few
unsourced judgements are marked `[impression]`.

One-line verdict: mistral.rs is a serving product with an engine inside it,
while bloomery is an engine with serving attached later. Its GGUF/CUDA stack is
serious prior art (llama.cpp-adapted fused kernels, per-arch GGUF bindings);
its CPU-offload MoE path is structurally broken on x86_64 in exactly the way
our measurements show. Copy the kernel strategy and the binding-table idea;
refuse the layer-granular device model, the trait-object token path, and the
per-step cache cloning.

## 1. Top-level structure: crates and the request path

### 1.1 The crates (15 workspace members, root `Cargo.toml`)

| Crate | Owns |
|---|---|
| `mistralrs-core` | Everything load-bearing: `Engine`, `Scheduler`, `Pipeline`s, all text `models/*`, KV cache, MLA, device maps, sampler |
| `mistralrs-quant` | All weight math: every `QuantMethod` (GGUF, GPTQ, HQQ, FP8, …), CUDA/CPU kernels under `kernels/`, `ShardedVarBuilder` |
| `mistralrs-server-core` | The axum serving layer: routers, OpenAI/Anthropic handlers, SSE streaming, tool dispatch surface |
| `mistralrs-cli` | The `mistralrs` binary: `serve`, `run`, `bench`, `quant`, `tune`, `doctor` subcommands |
| `mistralrs` | The Rust SDK: `*ModelBuilder`s wrapping `mistralrs-core` (`ModelBuilder`, `GgufModelBuilder`, …) |
| `mistralrs-paged-attn` | PagedAttention metadata + block manager (backend for the opt-in paged path) |
| `mistralrs-vision` / `-audio` | Multimodal preprocessors (image, audio) |
| `mistralrs-mcp` / `-macros` / `-code-exec` / `-sandbox` | Tool/agent surface: MCP client, tool proc-macros, code exec, subprocess sandbox |
| `mistralrs-metal-compile` / `-flash-attn` | Metal shader compilation; flash-attention kernel crate |
| `mistralrs-pyo3` | Python bindings |

The dependency direction is strict: everything funnels into `mistralrs-core`
+ `mistralrs-quant`; `mistralrs-server-core` and `mistralrs` are both thin
clients of `mistralrs-core`. The SDK's own architecture diagram
(`mistralrs/src/lib.rs:192-202`) states the shape:

```text
Model ──── send_chat_request() ──► Engine ──► Pipeline ──► Output
```

### 1.2 The path of one request, type by type

1. **HTTP entry.** `mistralrs-cli/src/commands/serve.rs:37` (`run_server`)
   builds an axum `Router` via
   `mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder`.
2. **Handler.** e.g. `mistralrs-server-core/src/chat_completion.rs`:
   `parse_request` turns the OpenAI JSON into a core `Request`, then
   `handler_core.rs:408` `send_request` → `send_request_with_model`
   (`handler_core.rs:415`) → `SharedMistralRsState::send_request_async`.
3. **Dispatch.** `mistralrs-core/src/lib.rs:2422`
   `MistralRs::send_request_async` picks the per-model engine sender and does
   `sender.send(request).await` over a tokio mpsc channel. `Request` is the
   enum in `mistralrs-core/src/request.rs` (782 lines).
4. **Engine loop.** `Engine::run` (`mistralrs-core/src/engine/mod.rs:1107`,
   ~1000 lines of loop body): admission queue (`admission::AdmissionQueue`)
   → frees finished sequences → `scheduler.schedule(...)`
   (`engine/mod.rs:1395`) → `SchedulerOutput::{completion, prompt}` batches.
5. **Pipeline step.** `Pipeline::step` / `submit_step`
   (`mistralrs-core/src/pipeline/mod.rs:1971/2002`, provided trait methods):
   `InputsProcessor::process_inputs` tokenizes/pads → `forward_inputs`
   (`pipeline/mod.rs:1782`) on the concrete pipeline (e.g. `GGUFPipeline`,
   `pipeline/gguf.rs:134`) → model `forward` → logits.
6. **Sampling.** `sample_and_add_toks` →
   `sample_sequence` (`pipeline/sampling.rs:1381`) → `Sampler`
   (`core/src/sampler.rs`, 2897 lines: temperature/top-k/top-p, grammar/llguidance,
   dry, XTC, …) → token appended to `Sequence`, `Response` sent back over the
   per-request channel; server-core renders SSE or JSON.

Key types on the hot path: `Request` → `Sequence`
(`sequence.rs`, 3724 lines, always behind `Arc<std::sync::Mutex<Sequence>>`)
→ `SchedulerOutput` → `Box<dyn Any>` inputs → `ForwardInputsResult`
(`pipeline/mod.rs:1564`, a 6-variant enum: `RawLogits`, `CausalGeneration`,
`Embeddings`, `Image`, `Speech`, `BlockGeneration` — every step matches on
modality) → `Response` (`response.rs`, 563 lines).

### 1.3 Which crate would we rewrite for the serving layer only?

`mistralrs-server-core` (axum handlers, OpenAI/Anthropic mapping, SSE) plus
the `serve` subcommand in `mistralrs-cli`. That is the whole serving surface;
`mistralrs-core` exposes `MistralRs::send_request_async` + a `Receiver<Response>`
as its complete API boundary (`handler_core.rs:400-432`), so a from-scratch
server against that boundary is self-contained. `[impression]` It is also the
part with the least to learn from: OpenAI mapping plus streaming plumbing,
no scheduling intelligence of its own.

## 2. Model loading: GGUF file to weights

### 2.1 The pipeline, function by function

For a GGUF text model the entry is `GGUFLoader::load_native_normal`
(`mistralrs-core/src/pipeline/gguf.rs:566`):

1. `mistralrs_quant::GgufArchive::open(weight_files)` (`gguf.rs:579`) —
   parses (mmaps) the GGUF container. Container parsing lives in
   `mistralrs-quant/src/gguf/archive.rs` (2127 lines).
2. Read `general.architecture` from metadata (`gguf.rs:582-588`); build a
   `GgufDescriptor { architecture, metadata_keys, tensor_names }`
   (`gguf.rs:624`, type in `gguf/normal_registry.rs:156`).
3. `resolve_native_adapter(&descriptor, explicit_loader)` (`gguf.rs:640`) —
   maps the GGUF arch string + tensor inventory to a `NormalLoaderType`
   (the HF-side model family). Special cases exist, e.g.
   `deepseek2_identity_hint` (`normal_registry.rs:1027`) disambiguates
   DeepSeek-V2/V3/R1 by `general.name`/`general.basename`.
4. Config: either normalize a sidecar HF `config.json`
   (`normalize_external_normal_config`, `gguf.rs:659`) or
   `synthesize_normal_config(&loader_type, metadata, tensor_names)`
   (`gguf.rs:667`, defined `gguf/normal_config.rs:153`) — i.e. they
   **fabricate the HF config JSON from GGUF metadata** so the rest of the
   load path is identical to the safetensors path.
5. `build_normal_bindings(&archive, &loader_type, architecture)`
   (`gguf.rs:670`, defined `gguf/normal_bindings.rs:7`) — the tensor-name →
   module mapping (see §2.2).
6. `GgufWeightSource::new(archive, &bindings, dtype)` (`gguf.rs:672`) +
   `source.sharded_var_builder(Device::Cpu)` (`gguf.rs:677`) — a lazy
   `ShardedVarBuilder` that materializes each native name on demand.
7. `NormalLoaderBuilder::build_with_source(loader_type, source, ...)`
   (`gguf.rs:720`) → `loader.load_model_from_path(...)` (`gguf.rs:721`) —
   from here the GGUF model loads through **the same `NormalLoader` as a
   safetensors checkpoint**. The GGUF/HF duality is resolved once, at the
   weight-source boundary, not threaded through the model code. This is the
   single best structural idea in their loader.

### 2.2 Where the tensor-name mapping lives

`mistralrs-core/src/gguf/normal_bindings.rs` (987 lines):
`build_normal_bindings` walks every tensor in the archive, parses
`blk.{layer}.{role}.{suffix}` via `parse_block_tensor`, and inserts
`GgufTensorBinding`s keyed by **HF native name** (`model.layers.{i}....`).
Root tensors are bound explicitly, e.g. (`normal_bindings.rs:46-69`):

```rust
bind(archive, bindings, "model.embed_tokens.weight", "token_embd.weight");
...
bind(archive, bindings, "lm_head.weight", "output.weight");
```

Composite layouts (stacked experts, split MLA `kv_b`, fused QKV slices,
rope permutations) are expressed as a small binding algebra in
`mistralrs-quant/src/gguf/weight_source.rs:392-462`
(`Tensor | Slice | Concat | Stack { inputs, dim } | Interleave | Transpose |
Reshape | Affine | ...`), materialized lazily in `materialize_binding`, with
the quantized-preserving branch in `load_linear` (`weight_source.rs:769`):
raw dtypes 0/1/30 (F32/F16/BF16) go dense (`load_dense_linear`), everything
else stays a `QTensor` (`load_direct_linear` / `load_structural_linear`).

### 2.3 Adding a new architecture — the full touch list

1. `mistralrs-core/src/models/<arch>.rs` — the model struct + `forward`
   (e.g. `deepseek2.rs`, 1121 lines).
2. `NormalLoaderType` variant — `pipeline/loaders/normal_loaders.rs:178`
   (26 variants today).
3. `<Arch>Loader` implementing the loader traits in `normal_loaders.rs`
   (the file is 7335 lines; `DeepSeekV2` appears 28 times).
4. Match arm mapping variant → loader — `pipeline/normal.rs:511-536`.
5. `CanonicalGgufArchitecture` variant + `GgufSchema` (required
   metadata/tensors, rope pairing, layouts) — `gguf/normal_registry.rs`
   (1472 lines; 26 canonical archs).
6. Binding arms — `gguf/normal_bindings.rs` (per-loader `match`es, e.g.
   `bind_deepseek_kv_b` at `:472`).
7. Config synthesis arms — `gguf/normal_config.rs` (3472 lines).

Per-arch code is thus scattered across **six files plus the model file**,
coordinated by three parallel enums (`NormalLoaderType`,
`CanonicalGgufArchitecture`, `GgufLayout`). The shared substrate it all
leans on is large: `layers.rs` (4241 lines: norms, `ColumnParallelLayer`,
`ReplicatedLayer`, activations), `ops.rs` (8620 lines), `moe/experts/*`
(~2700 lines), `mla/*` (~1400 lines), `attention/*`.

### 2.4 Comparison with llama.cpp

llama.cpp's DeepSeek-V2 support is **one 713-line file**,
`src/models/deepseek2.cpp`: `load_arch_hparams` (metadata → hparams, :3),
`load_arch_tensors` (tensor-name mapping, :55), `build_arch_graph` (the
forward graph, :163), with tensor names drawn from the shared
`LLM_TENSOR_*` enum table. One arch = one file + rows in shared tables.
mistral.rs's equivalent is one 1121-line model file **plus coordinated edits
in six registry/binding/config files**. The cost of their approach is visible
in `deepseek2_identity_hint` (`normal_registry.rs:1027-1044`): string
sniffing on `general.name` to tell V2 from V3/R1, needed because the GGUF
arch string alone does not select the loader. llama.cpp never has this
problem class — the arch enum comes from the same file's loader, and tensor
presence is checked per-tensor at load.

## 3. Device placement and dtype

### 3.1 Who decides the device: `DeviceMapper`, indexed by layer, nothing else

The entire placement vocabulary is the trait in
`mistralrs-core/src/device_map/mappers.rs:9-26`:

```rust
pub trait DeviceMapper: Debug {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;
    fn set_device(&self, layer: usize, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder;
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    ...
}
```

Every method takes `layer: usize`. The production impl, `LayerDeviceMapper`
(`mappers.rs:29`), holds `mappings: Vec<Device>` — **one device per
repeating layer** — and `map` is just
`self.cuda_peer_access.to_device(&input, &self.mappings[layer])`
(`mappers.rs:50-53`): move the activation to layer `i`'s device, no dtype
touch. Model code calls `mapper.set_device(layer_idx, vb.pp(...), ...)` for
every submodule at build (`models/deepseek2.rs:177-257`, dozens of
call sites) and `xs = self.mapper.map(xs, i)?` per layer per token at
forward (`models/deepseek2.rs:969`).

The `mappings` vector itself is built in `device_map/mod.rs:87-185` from
`DeviceMapMetadata { device_layers: per-ordinal layer counts, host_layers }`
— counts of layers per GPU plus a host-layer count — or from a `Topology`
that is still a per-layer `Vec` (`mod.rs:132-141`). The `Auto` path
(`pipeline/loaders/auto_device_map.rs:200` `get_device_layers`) likewise
returns `DeviceMapMetadata`: layer counts that fit VRAM. Non-layer tensors
(embeddings, norm, lm_head) go to the single `nm_device`.

There is exactly one per-tensor-name hook in the whole system,
`get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor>`
(`utils/varbuilder_utils.rs:77`), but its codomain is
(`varbuilder_utils.rs:56-59`):

```rust
pub enum DeviceForLoadTensor {
    Base,
    Idx(usize),
}
```

and the `NormalLoader` impl (`pipeline/loaders/normal_loaders.rs:150-171`)
parses the layer index out of the name with the regex
`\.layers\.(\d+)\.`; anything without a layer number maps to `Base`.
So the hook can see a tensor name but can only say "base device" or "layer
N's device".

**Answer: per-tensor placement is absent by design.** The `DeviceMapper`
interface cannot express it (layer-indexed), the metadata that feeds it is
layer counts, and the one name-aware hook collapses names back to layer
indices. There is no `-ot`-style override because there is no address in the
system to attach one to — adding it would mean changing the trait, the
metadata, the auto-mapper, and every model's `set_device` call pattern.

### 3.2 Who decides the dtype: one global `DType` for the whole model

`TryIntoDType::try_into_dtype(&self, devices: &[&Device])`
(`utils/normal.rs:55`) returns a single `DType`. `ModelDType::Auto`
(`normal.rs:171`) calls `determine_auto_dtype_all` (`normal.rs:121`), which:
returns F32 if all devices are CPU (`normal.rs:131-133`); otherwise shells
out to `nvidia-smi` for minimum compute capability (`normal.rs:78-96`) and
probe-matmuls BF16/F16 on every device, falling back to F32. One dtype, whole
model, no per-layer or per-tensor dtype anywhere downstream — `map()` moves
devices without casting, and `GgufWeightSource` is constructed with the
single `internal_dtype` (`pipeline/gguf.rs:671-676`).

### 3.3 The structural reason for our BF16-vs-F32 offload failure

Chain it end to end for "one MoE layer on the host, rest on CUDA":

1. Global dtype resolves to BF16 (CUDA probe passes).
2. Layer `i`'s activation arrives on CPU still BF16 (`map` doesn't cast).
3. The CPU GGUF MoE entry `cpu_indexed_moe_forward`
   (`mistralrs-quant/src/gguf/cpu.rs:94`) → `qtensor_indexed_moe_forward`
   (`cpu.rs:23`) first tries the packed path `qtensor.indexed_gemv(...)`
   (`cpu.rs:61`) — which **always returns `None` on x86_64** (see §4.4).
4. It falls through to (`cpu.rs:74-80`):

```rust
// Dequantize all weights to f32
let weights = qtensor.dequantize(device)?;
let unquant = UnquantLinear::new(...)?;
unquant.gather_forward(x, ids)
```

Weights are now F32, `x` is still BF16, and `UnquantLinear`'s
`quantized_act_type()` is `None` (`unquantized/mod.rs:371-373`), so the
`QuantMethod::gather_forward` default (`quant/src/lib.rs:1813-1821`) skips
its cast wrapper and calls `gather_forward_raw` directly — dense
`index_select` + `matmul` (`unquantized/mod.rs:251-290`), where candle
rejects the BF16×F32 matmul. **The assumption is made at `cpu.rs:75`: the
fallback assumes the activation is already F32.** Note the dense path does
not share the bug: `GgufMatMul::forward_raw`'s fallback explicitly casts
activation to F32 and back (`gguf/mod.rs:465-478`, comment: "Fallback: Candle
QMatMul requires F32"). Our PR #2430 ports exactly that two-line cast
discipline to the MoE gather fallback.

The deeper structural point: nothing in the load path *could* have prevented
this, because dtype is global while kernel dtype requirements are per
(device, path) — the information "this CPU fallback needs F32" exists only
as a comment at the call site. `[impression]` Every new (device × quant
path) combination re-rolls these dice.

## 4. Quantized matmul: provenance, dots-vs-dequantize, and MoE gather

### 4.1 Where the kernels come from: a three-way mix

| Path | Kernels | Provenance |
|---|---|---|
| Dense CUDA, batch 1–8 (`fast_mmvq::plain`) | `kernels/mmvq_gguf/mmvq_gguf.cu` | Their own, "**Adapted from llama.cpp's CUDA mmvq path**" (file header, :1-2), Q8_1 activations |
| Dense CUDA, batch >8 (`fast_mmq::plain`) | `kernels/mmq_gguf/*` | Their own, same llama.cpp lineage (mmq/mmq_vecdot file names) |
| Dense CUDA, large batch (`packed_affine`) | `kernels/gguf_affine_packed/*`, `marlin/*` | Marlin-style repack; gated on `has_marlin_kernels` build flag |
| Dense CUDA fallback | candle `QMatMul::forward` with F32 cast | candle (`gguf/mod.rs:465-478`) |
| Dense CPU | candle `QTensor` custom op `cpu_fwd` | candle: real quantized `matmul_t` vec_dots (`k_quants.rs`, ggml ports) with AVX2/FMA dispatch (`has_avx2_fma`, `k_quants.rs:15-17`, `avx.rs`), or the `repack` packed path |
| MoE CUDA decode | `kernels/indexed_moe/indexed_moe.cu` | Their own, "**Adapted from llama.cpp ggml-cuda.cu and candle-kernels**" (file header, :1-3) |
| MoE CUDA prefill | grouped path (`moe_dispatch_build`, `gguf/cuda.rs:590`) + CUTLASS (`kernels/cutlass_moe/*`, `moe/cuda.rs`) | Their own dispatch + CUTLASS |
| MoE CPU/Metal | candle `QTensor::indexed_gemv` **or** dequantize-all + dense bmm | candle packed path is **aarch64-only**; on x86_64 it is always the fallback (`gguf/cpu.rs:23-81`) |

Dispatch order for dense CUDA is fixed in `try_fast_forward`
(`gguf/mod.rs:299-323`): batch ≤ 8 → MMVQ, batch > 8 → MMQ, else the F32
fallback; `packed_affine_forward` is tried before all of it when the Marlin
build flag is on (`gguf/mod.rs:440-447`). `MMVQ_MAX_BATCH = 8`
(`fast_mmvq.rs:53`).

### 4.2 Dequantize-then-GEMM vs real quantized dots

Real quantized dots almost everywhere — with one giant exception:

- **CUDA dense/MoE**: true quantized kernels with Q8_1-quantized
  activations on the fly (`mmvq_gguf.cu` header: "GGUF matvec kernels with
  Q8_1-quantized activations"; MoE entry
  `indexed_moe_forward_fused_q8_1_input`, `gguf/cuda.rs`).
- **CPU dense**: candle `CustomOp1 for QTensor::cpu_fwd`
  (`quantized/mod.rs:1169`) calls `repack::try_matmul_f32` then
  `matmul_t` — the ggml `vec_dot` ports. BF16 activations are widened to
  F32 once, output narrowed back (`mod.rs:1229-1250`).
- **CPU MoE on x86_64**: **dequantize all 64 experts to F32, then dense
  `index_select` + bmm** (`gguf/cpu.rs:74-80` +
  `unquantized/mod.rs:263-290`). Not a dot product in sight.

### 4.3 How the 6-of-64 routed experts are gathered and multiplied

Model side: `Moe::forward` (`models/deepseek2.rs:665`) routes
(`MoeGate::forward` → top-k indices + weights) then calls
`MoEExperts::forward(xs, topk_weight, topk_idx)` (`deepseek2.rs:672`), where
`MoEExperts` (`moe/experts/mod.rs:37`) is a backend enum
(`moe/experts/mod.rs:49`: `Fused | Cutile | CutileFp8 | Cutlass | Fast`).
The GGUF decode path is `forward_gather` (`moe/experts/backends.rs:1163`):

```rust
let gate = self.fused_gate_proj.gather_forward(&xs, forward.topk_ids)?;
let up   = self.fused_up_proj.gather_forward(&xs, forward.topk_ids)?;
...
       .gather_forward(&down_in, forward.topk_ids)?
```

(`backends.rs:1179-1184`). Expert weights are pre-stacked at load to
`[n_experts, out, in]` via the `Stack` binding (`weight_source.rs:98,
:424`) — the `StackedExperts` layout (`normal_registry.rs:127`). Each
`gather_forward` on CUDA becomes one fused `indexed_moe_forward_<qtype>_q8_1`
kernel launch over (token, expert-id) pairs (`gguf/cuda.rs:440-480`
per-dtype launch table); prefill uses the grouped path with explicit
dispatch tables (`moe_dispatch_build`, `gguf/cuda.rs:590`).

### 4.4 Why x86_64 CPU MoE is the exception — and why it costs 442 ms

candle's `QTensor::indexed_gemv`
(`candle-core/src/quantized/mod.rs:1046-1121` at the pinned rev) is
gated on `#[cfg(target_arch = "aarch64")]`; the non-aarch64 branch is:

```rust
let _ = (x, ids);
Ok(None)
```

(`mod.rs:1116-1120`). So on our Threadripper, mistral.rs's
`if let Some(out) = qtensor.indexed_gemv(&x3, &ids2)?` (`gguf/cpu.rs:61`)
can never hit, and **every MoE token on the host dequantizes all 64 experts
of gate, up, and down projections to F32** (three full dequantizations of
tensors ~64× the routed working set), then `index_select`s 6 experts' rows
and runs dense bmm. Against ik_llama.cpp's 0.20 ms of sparse Q3_K×Q8_K
AVX2 dots over 6 experts, our measured 442 ms/layer/token is not a tuning
gap — it is dequantizing ~10× the needed bytes to F32 plus dense math at
F32 width. `[impression]` Even if `indexed_gemv` grew an x86 backend, the
repack cache it depends on (`repacked_qs`, `repack::PackedCache`) repacks
all experts eagerly; a sparse kernel over routed experts only (what our
`q3k-cpu` already is) would still win on memory traffic.

Note the docs-vs-code wrinkle: `GgufMatMul::quantized_act_type`
(`gguf/mod.rs:560-565`) comments "cpu handles bf16 activations natively
(widened once inside the packed matmul)" — true of the dense CPU path, false
of the CPU MoE fallback two files away. A reader trusting the comment would
never predict the BF16/F32 failure.

## 5. KV cache and attention

### 5.1 Layout and allocation: contiguous buffers, grown 512 tokens at a time

The default (non-paged) cache is `NormalCache(Vec<KvCache>)`
(`kv_cache/mod.rs:415`), one `KvCache` per layer, each `{ k, v }` a
`SingleCache` (`mod.rs:50-54`) — or `RotatingCache` for sliding-window
models, or `Shared { owner }` for cross-layer sharing. `SingleCache`
(`kv_cache/single_cache.rs:11-21`) is a lazily allocated contiguous
`all_data: Option<Tensor>` with `current_seq_len`; `append`
(`single_cache.rs:102-137`) allocates `[..., capacity, ...]` zeros on first
use and, when full, grows `capacity_seq_len` by `CACHE_GROW_SIZE = 512`
(`mod.rs:426`), reallocates, and copies the old contents over
(`single_cache.rs:114-130`). No paging: one contiguous allocation per
k/v per layer, realloc-and-copy growth. CPU KV defaults to F16 where AVX2
+ FMA + F16C (or AVX512) is present (`cpu_kv_f16`, `mod.rs:70-96`;
+ overridable via `MISTRALRS_CPU_KV_F32=1`).

Paged attention exists but is opt-in and separate: `PagedAttention` /
`PagedAttentionConfig` (`paged_attention/mod.rs`, 558 lines) backed by the
`mistralrs-paged-attn` crate (block manager, metadata), enabled per model
(`paged_attn: Option<PagedAttention>` in `models/deepseek2.rs`), with a GPU
KV budget (`MemoryGpuConfig`, `auto_device_map.rs`). The engine threads
`CacheBackendMetadata::{DefaultInstructions, PagedAttention}` through every
step (`pipeline/mod.rs:1553-1561`).

### 5.2 `deepseek2` support: yes, and MLA has three paths

`mistralrs-core/src/models/deepseek2.rs` (1121 lines) implements
DeepSeek-V2-Lite-style MLA + MoE; the GGUF registry maps `"deepseek2"`
(`normal_registry.rs:95`) and the loader enum carries `DeepSeekV2`
(`normal_loaders.rs:200`). `Attention::forward` (`deepseek2.rs:287-430`)
branches three ways:

1. **Fused MLA decode** — `mla_decode_forward` (`deepseek2.rs:337`), using
   FlashInfer MLA kernels over the compressed latent. Gate
   (`mla/forward.rs:88-106`): unmasked single-token decode **on CUDA with
   paged attention and FlashInfer metadata**; the non-CUDA build's version
   unconditionally returns `false` (`mla/forward.rs:109-118`).
2. **MLA cache (prefill with prefix caching)** — `mla_cache_forward`
   (`deepseek2.rs:384`). Gate (`mla/forward.rs:128-137`): paged attention
   **on CUDA**. Non-CUDA: `false` (`mla/forward.rs:140-146`).
3. **Expanded/absorbed fallback** — everything else, including all CPU
   execution: either weight-absorbed attention over the repeated latent
   (`use_absorbed`, `deepseek2.rs:355-362`, requires split `kv_b` weights —
   the `SplitMlaKvB` GGUF layout, `normal_registry.rs:128`,
   `bind_deepseek_kv_b`, `normal_bindings.rs:472`) or full `expanded_kv`
   per-head K/V materialization (`deepseek2.rs:364`) followed by the
   generic attention dispatch (`attention/mod.rs`, flash-with-`is_causal`
   or eager fallback).

So on a CPU-offloaded layer, MLA never touches a fused kernel: it attends
over either a head-repeated latent or fully expanded K/V. `[impression]`
For stage 1 (CPU-first, M=1) this fallback structure is actually the right
shape to copy — absorbed over the latent, no expansion — just without the
two CUDA-only paths above it.

## 6. What their abstractions cost — four concrete sites

**1. The pipeline and scheduler are `Mutex`-guarded trait objects, locked
twice per step.** `Engine` (`engine/mod.rs:200-207`):

```rust
pipeline: Arc<Mutex<dyn Pipeline>>,
scheduler: Arc<Mutex<dyn Scheduler>>,
```

Every engine iteration locks the scheduler to schedule (`engine/mod.rs:1391,
1395`) and locks the pipeline separately for the completion batch
(`engine/mod.rs:1409`) and the prompt batch (`engine/mod.rs:1473`). Dynamic
dispatch on every step call plus mutex acquisition on the token path — for a
single-threaded decode loop that has exactly one pipeline and one scheduler
for the life of the process.

**2. Forward inputs are type-erased: `Box<dyn Any>` per step.**
`Pipeline::forward_inputs(&mut self, inputs: Box<dyn Any>, ...)`
(`pipeline/mod.rs:1782-1786`); every `step` boxes, every `forward_inputs`
downcasts (`submit_step`, `pipeline/mod.rs:2020-2034`), and the result is
re-wrapped in the 6-variant `ForwardInputsResult` enum and matched on
modality (`pipeline/mod.rs:2149-2243`). The type system knows the model is a
causal LM; the call boundary forgets it on every token.

**3. Every matmul is a virtual call: `Arc<dyn QuantMethod>`.**
Model structs hold e.g. `kv_a_proj_with_mqa: Arc<dyn QuantMethod>`
(`models/deepseek2.rs:144`) and `Plain(Arc<dyn QuantMethod>)`
(`deepseek2.rs:125`); `forward`/`gather_forward` dispatch through the
`QuantMethod` vtable (`quant/src/lib.rs:1795-1821`) per projection per layer
per token, on top of the per-layer `Box<dyn DeviceMapper>` virtual `map`
call (`deepseek2.rs:969`, mapper stored as
`mapper: Box<dyn DeviceMapper + Send + Sync>`, `pipeline/gguf.rs:143`).

**4. The default cache path memcpys the whole cache twice per step.**
`NormalCacheManager::clone_in_cache` (`kv_cache/mod.rs:486-560`) allocates
fresh `Tensor::zeros` batched k/v per layer and `slice_set`s every
sequence's cache into it; `clone_out_cache` (`kv_cache/mod.rs:634+`)
scatters it back after the step. The comment at `mod.rs:499` says
"Preallocate combined k and v caches across all sequences, avoiding
Tensor::cat copies" — the copies were optimized from cat to slice_set, but
the clone-in/clone-out round trip per step remains structural: caches live
on `Sequence`s, compute wants them batched on the pipeline. (The older
`Cache::update_kv_cache`, `kv_cache/mod.rs:922-936`, still
`Tensor::cat`s the full history per token per layer — O(n²) total copies —
and is still live in 12 files: all of `xlora_models/*` and the llava
`vision_models`, per repo code search.)

## 7. What bloomery should copy, and what it should refuse

Licenses first, as specified. mistral.rs: **MIT**, `Copyright (c) 2024 Eric
Buehler` (root `LICENSE`, 1068 bytes: "MIT License … Permission is hereby
granted …"). bloomery: **MIT**, `Copyright (c) 2026 midagedev` (root `LICENSE`).
MIT-to-MIT copying requires preserving the upstream copyright + permission
notice in the copies — that is the whole attribution burden. (Their CUDA
kernels adapted from llama.cpp inherit llama.cpp's MIT the same way.)

### Copy (idea only — bloomery's kernels are hand-written and faster; do not paste)

1. Resolve the GGUF/HF duality once at the weight-source boundary so model code never branches on container (`pipeline/gguf.rs:672-729`: `GgufWeightSource` → shared `NormalLoader`).
2. Express composite weight layouts (stacked experts, split MLA `kv_b`, fused QKV) as a small lazy binding algebra rather than load-time special cases (`weight_source.rs:392-462`).
3. Ship fused decode/prefill kernel pairs per (quant, device) with an explicit batch-size crossover instead of one kernel for both (`try_fast_forward`, `gguf/mod.rs:299-323`, `MMVQ_MAX_BATCH = 8`).
4. Keep the CPU KV cache in F16 behind a runtime feature probe with an env-var escape hatch (`cpu_kv_f16`, `kv_cache/mod.rs:70-96`).
5. Synthesize the runtime config from GGUF metadata (tensor inventory as validation input) so the file is self-describing (`synthesize_normal_config`, `gguf.rs:667` + `normal_config.rs:153`).

### Refuse

1. Refuse layer-granular placement as the only address space — bloomery's stage 2 needs per-tensor (expert) placement, which their `DeviceMapper` trait cannot express (`mappers.rs:9-26`).
2. Refuse a single global dtype for mixed-device execution — per-(device, path) kernel dtype requirements must be decided where the kernel is chosen, not probed once at startup (`utils/normal.rs:121-169` vs `gguf/cpu.rs:75`).
3. Refuse `Mutex<dyn Trait>` + `Box<dyn Any>` on the token path — one engine, one model, monomorphized dispatch (`engine/mod.rs:200-207`, `pipeline/mod.rs:1782`).
4. Refuse per-step cache clone-in/clone-out — caches should live where compute consumes them, appended in place (`kv_cache/mod.rs:486-560`, `:634+`).
5. Refuse six-file architecture registration — one arch, one file, tables for names, like llama.cpp's 713-line `src/models/deepseek2.cpp` instead of the `NormalLoaderType` + `CanonicalGgufArchitecture` + bindings + config sprawl (§2.3).

## 8. The PR (#2430)

- **What it changes:** `mistralrs-quant/src/gguf/cpu.rs`, +5/−2, one commit
  (`b0f26d5c`, "fix(quant): accept non-F32 activations in the CPU GGUF MoE
  gather"): in `qtensor_indexed_moe_forward`'s dequantize fallback, cast `x`
  to F32 before `UnquantLinear::gather_forward` and cast the result back to
  the input dtype — mirroring the dense fallback's cast discipline at
  `gguf/mod.rs:465-478`.
- **State:** OPEN, created 2026-09-17 by midagedev, no human reviews (0),
  no requested changes; the only comment is the github-actions Code Metrics
  bot. `mergeable: MERGEABLE`, `mergeStateStatus: UNSTABLE` (clean merge,
  checks not all green).
- **Tree movement:** none. The PR's base (`d5ae0f18`) is identical to current
  `master` HEAD (compare: `identical`, 0 commits ahead/behind), and the
  touched file was last changed 2026-07-07 (#2311). It still applies cleanly.

## What I would put in bloomery's stage-1 spec because of this

1. Give every tensor a (device, dtype) address at load time — never a per-layer device plus a global dtype.
2. Put each model's tensor-name table and forward in one file, with shared tables for names, not per-arch match arms across registry files.
3. Resolve GGUF-vs-anything-else at the weight-source boundary so no model code branches on container format.
4. Specify the MoE CPU path as sparse-dots-over-routed-experts from day one, with a gate that fails if any fallback dequantizes unrouted experts.
5. Keep stage-1 dispatch monomorphic (one model, one engine, no trait objects) and measure the cost before adding any.
6. Append KV in place in preallocated buffers with block growth, and forbid clone-in/clone-out batching on the decode path.
7. Ship MLA in weight-absorbed form over the latent on CPU first, and treat any fused/CUDA MLA kernel as a later optimization, not the design.



