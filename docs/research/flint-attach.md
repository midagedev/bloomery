# Flint: reader and common-owner attachment design (#632)

Design snapshot: **`e53cabebbb9b16249c81b7c26a026da1670f7d9d`**, requested by
[#632](https://github.com/midagedev/bloomery/issues/632). The source-gap observations below describe
that snapshot. The current tree additionally has a **header-only reader**; it still has no Flint
body, app session or serve seat. No GPU, inference, performance or weight-content validation is claimed.

## Host reader status

`arch/flint` reads independent bool SWA/RoPE masks, explicit head widths, the dense lead,
sigmoid router facts, optional selection bias and one explicitly sized ungated shared MLP.
It reuses the common metadata validators, tensor stems, classification and dimension checks.
`arch::per_layer_bool` owns exact-length bool-array validation for both Flint masks and
GemmaEmbedding2's sliding mask: missing keys, scalars, wrong lengths and non-bool items are refused.
Present `expert_weights_scale` uses `arch::finite_positive`; only an absent key defaults to 1,
with that default recorded.
The common needs walk emits q/k normalization even on NoPE layers. `program_of(Flint)` remains
`None`; missing program, head128 sinks/SWA, NoPE normalization and 256/top6 bindings remain coverage
refusals. `Router.norm` and `RouterRule.norm` distinguish no normalization, the existing selected-sum
rule and Flint's selected-sum floor; the floor description does not supply a device implementation
or activated-value scaling.

Fault formatting uses `bloomery_device::fault`'s shared layout/decoder and the
`gpu::arch::step_order` registry. Flint's registry row uses neutral code order;
it does not describe a runnable step.

Run `BLOOMERY_BOX_ENV="BLOOMERY_MODEL=/absolute/path/flint-header.gguf" BLOOMERY_BOX_LOCAL=1 just gate-flint-meta`.
Set `BLOOMERY_ENV_FILE` to the local toolchain environment file if it is not `~/bloomery-env.sh`.
Keep the outer `BLOOMERY_MODEL` unset or a tool-profile name: `box.sh` reads it before launching the command.
The model-path assignment belongs in `BLOOMERY_BOX_ENV`, which reaches the test after profile resolution;
that channel accepts paths without whitespace.
The gate hashes the first 5,470,528 bytes before using `gguf::inventory_of`, `write::Layout`
and `Split::header_only`; it reads no tensor payload and never treats a sparse file as real weights.
The pinned header yielded 779 tensors, 49 layers, 2 dense/47 routed blocks, 49 head-normalization
needs, 35 rotary needs and 49 sink-attention needs. An external reader smoke also exercised
short/malformed masks, zero window, nonfinite scale, missing sink/norm/shared tensors, a malformed
optional bias and an unknown tensor: each was refused with the offending key/name.

The independent common Jinja and tokenizer changes are tracked by #632; a reader does not imply
those changes are present in an older checkout. Fixture/refset registration is intentionally absent:
the artifact license/provenance conflict below is unresolved, and runtime support still requires the
remaining acceptance rows. This header gate is not an end-to-end model gate.

### Runtime handoff and host policy

The design, common template support, tokenizer/four-EOG contract and reader (#676–#680) are merged.
The existing header recipe was exercised on the exact 5,470,528-byte prefix above: four reader tests and
the pinned-header test passed, still reporting the missing program and operation bindings.
No full weights or GPU were needed for that check.

The maintainer [assigned the model port and offered card runs](https://github.com/midagedev/bloomery/issues/632#issuecomment-6352066886),
and named a forthcoming head128 sink/SWA attention, QK RMS and sigmoid256/top6 train.
Those complete contracts are not in this main snapshot or the inspected public heads of #726 (`1723485602c1`)
and #747 (`f883e84f8d2c`). Existing ring and sink-merge ingredients do not establish the combined attention
operation; `route_body` still lacks 256/top6 and its normalization-floor policy.
Obtain the train's fetchable ref before duplicating those primitives.

The router description/selection contract now distinguishes `RouterNorm::None`, `SelectedSum` and
`SelectedSumFloor`. Flint requests `max(sum_selected, 2^-14)`; existing model descriptions and arithmetic
keep their rules. Compiled rows, including boolean `norm_arg` bindings, refuse the floor policy rather
than borrow an epsilon-added implementation. The absent-bias, 256/top6 and `Need::Program` refusals stay
until real device bindings are supplied. Reconcile the policy interface with the pending train before device work.
Then close shared IQ4_XS, selected card/tier down forms and six-slot F32 handoff in their existing common owners,
followed by paging/resident slots, reference provenance and the body/app/seat.
Each device slice needs its numerical/fault gates and the maintainer's offered hardware proof before admission.
Legacy metadata booleans share `RouterNorm::from_legacy_bool`: false keeps scores and true uses the
existing epsilon-added selected sum. Flint's distinct floor mapping stays in its reader.

### Six-slot F32 tier handoff implementation

`HandoffKernels::enqueue_handoff_tier` now selects a genuine six-slot F32 entry
beside the existing eight-slot entry. Both host and tier layouts must have the
same supported slot count and activation width; compressed tier images remain
refused. The entry uses the existing copy/place/fault helpers and the existing
subsequent go/publication protocol; no Flint router or Program is registered.

The pure layout contract passed locally after an oxide build.
`hw_f32_tier_handoffs_conserve_six_and_eight_consumed_routes`, owned by
`gate-gpu-ds41-lib`, consumes the handoff's weights and emitted places through
the production tier accumulator and adds the host sum. It covers partial widths,
invalid ids, poisoned non-owning rows, trailing sentinels and changed replay
weights. Its host sum is precomputed and activation images are checked by
readback: it is not expert execution or two-card completion/publication proof.
The hardware case has not been run here.

The new `ffn_handoff_6_tier` entry needs add-class PTX/JIT/resource and
owning-gate evidence; the eight-slot and other existing entries must remain
unchanged. No PTX resource pin is added without an observed NVIDIA JIT run.
Actual six-slot body/tier and two-card completion remain integration work, not
capabilities inferred from this primitive.
`check-ptx-names` remains red until observed spill/JIT-local measurements can pin
the new entry in the four existing architecture/capability tables. Offline
no-JIT scans cannot supply that missing column, so no placeholder row is added.




## 1. Sources and observed artifact

- Official [config](https://huggingface.co/RecursiveSI/flint/raw/2a247be0c0aec895ccb0ac7ca92229ec6df37288/config.json),
  [model card](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/README.md) and
  [chat template](https://huggingface.co/RecursiveSI/flint/raw/2a247be0c0aec895ccb0ac7ca92229ec6df37288/chat_template.jinja):
  RecursiveSI/flint **`2a247be0c0aec895ccb0ac7ca92229ec6df37288`**.
- Actual GGUF specimen: unsloth/flint-GGUF **`20468edfc38ab43d6914cc88b08cbecb05d0fc62`**,
  [flint-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/flint-GGUF/blob/20468edfc38ab43d6914cc88b08cbecb05d0fc62/flint-UD-Q4_K_XL.gguf).
  The Hub lists 25,054,688,704 bytes. Its LFS digest is provenance from the Hub, **not** a locally verified full-file digest.
- Numerical/reference contract: mainline llama.cpp [Flint implementation](https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/models/flint.cpp)
  and [converter](https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/conversion/flint.py),
  merge **`a37cc46104eb50900d5ab4a9ba02a0abdac039db`** of [#30928](https://github.com/ggml-org/llama.cpp/pull/30928).
  Mainline is the oracle; an ik build without Flint is not an alternative oracle.
- [Measured header evidence](flint-attach-evidence.json) records pins, hashes, metadata, shapes and exact
  per-role/per-layer quant types. All shape axes below are GGUF `ne[]`, contiguous axis first.

The existing `tools/bloomery/hf.py` range reader fetched an 8 MiB prefix; the header ends, including
alignment padding, at byte **5,470,528**. Bytes after that boundary were ignored. Header SHA-256:
`fb0cc4b520f927f0b679193397ee0c36ca252426e7881283c5aa6de3e44a3a2b`.
The existing Python header reader and Rust `gguf::inventory_of` agree on **779 tensors** and
**25,049,218,176 encoded tensor bytes**. Shape products total **39,785,287,456 parameters** (derived).
No full model was downloaded or executed.

The official repository says Apache-2.0; this exact GGUF header says `other` / `flint-preview`.
The config/header shapes and the byte-identical 2,328-byte chat templates establish those contracts,
**not equivalence of the weights or licenses**. Resolve the artifact's provenance/license with its
publisher before redistributing a fixture or describing this GGUF as an Apache-2.0 artifact. Do not
silently rewrite the header or make the engine's architecture reader a license checker.

## 2. File facts the reader must preserve

| Fact | Exact contract |
|---|---|
| Model | `general.architecture=flint`; width 2,048; 49 blocks; untied embedding/output; vocabulary 150,144 |
| Attention | 32 Q heads, 4 KV heads, head dimension 128; Q width **4,096**, not the residual width; RMS epsilon `1e-6` |
| Layer masks | Full attention and NoPE at **0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48**; the other 35 layers use SWA and RoPE |
| Window/position | 513 tokens = current token plus 512 predecessors; RoPE base 10,000; configured maximum context 262,144 |
| Dense FFN | Layers 0–1, SiLU-gated, intermediate width 6,144 |
| Routed FFN | Layers 2–48; 256 experts, top 6, each intermediate width 512; sigmoid scores; selected-weight normalization; scale 1 |
| Shared FFN | One always-active physical gate/up/down stack, intermediate width 1,024; GGUF `expert_shared_count=2` counts width units of 512, not two separately routed modules |
| QK norms/sinks | Per-head Q/K RMSNorm weights `[128]`, shared across heads, and one learned sink logit per Q head `[32]` in every layer |
| Tokenizer | `gpt2` BPE, `pre=flint`; 150,144 tokens and 149,744 merges; `add_bos_token=false`, `add_eos_token=false` |
| Control IDs | BOS 150000; PAD/EOS 150001; EOG set **150001, 150003, 150005, 150018** |

Both GGUF boolean arrays (`attention.sliding_window_pattern`, `attention.rope_pattern`) were read by
Rust and compared with the official config's `layer_types` / `no_rope_layer_indices`; they agree.
The initial two and final layers make a blind `layer % 4` rule wrong. Read and validate the arrays;
never reconstruct this model from the issue's shorthand “three sliding to one full.” The GGUF EOG
array also agrees with all four config EOS IDs, not only the scalar `eos_token_id`.

### Tensor roles

| Scope / GGUF suffix | Shape | Types actually observed in this specimen |
|---|---|---|
| `token_embd.weight`, `output.weight` | `[2048,150144]` | Q8_0, separate tensors |
| `output_norm.weight`; block `attn_norm.weight`, `ffn_norm.weight` | `[2048]` | F32 |
| Block `attn_q.weight` | `[2048,4096]` | Q8_0, Q5_K |
| Block `attn_k.weight`, `attn_v.weight` | `[2048,512]` | Q8_0, Q5_K |
| Block `attn_output.weight` | `[4096,2048]` | Q8_0, Q5_K |
| Block `attn_q_norm.weight`, `attn_k_norm.weight` | `[128]` | F32 |
| Block `attn_sinks.weight` | `[32]` | F32 |
| Dense `ffn_gate.weight`, `ffn_up.weight`; `ffn_down.weight` | `[2048,6144]`; `[6144,2048]` | Q8_0 |
| Routed `ffn_gate_inp.weight` | `[2048,256]` | BF16 |
| Routed `ffn_gate_exps.weight`, `ffn_up_exps.weight` | `[2048,512,256]` | IQ3_S, Q5_K, IQ4_XS, Q4_K |
| Routed `ffn_down_exps.weight` | `[512,2048,256]` | Q5_K, Q8_0, Q6_K, IQ4_XS |
| Shared `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`; `ffn_down_shexp.weight` | `[2048,1024]`; `[1024,2048]` | Q8_0, Q5_K, IQ4_XS |

There is **no router correction-bias tensor** in the 779 descriptors. In particular, gate/up expert
layer 2 is IQ3_S, expert-down layer 4 is Q8_0, expert-down layer 8 is Q6_K, and all three shared
projections in layer 42 are IQ4_XS. Admission must check each role and each layer; the filename's
`Q4_K_XL` is not a kernel capability. The companion evidence lists every layer/type association.

## 3. Numerical contract, not just matching dimensions

The [reference graph][ref-flint] is sequential pre-norm attention and FFN:
`a = x + Attention(RMS2048(x))`, then `y = a + FFN(RMS2048(a))`.
Q and K receive learned RMSNorm **within each 128-element head** before optional RoPE; V does not.
SWA layers use full-head NeoX pairs `(j,j+64)` for `j=0..63`; NoPE layers perform normalization
without rotation. Attention scales logits by `1/sqrt(128)`.

A learned sink is a **logit with no value vector**. For head `h`, visible scores `z_j` and sink `s_h`,
its output is `sum(exp(z_j-m)*v_j) / (exp(s_h-m)+sum(exp(z_j-m)))`, with
`m=max(s_h,max(z_j))`. Add the sink denominator **once**, not once per partial attention segment.
It consumes no KV row. A zero sink logit is not the same as an absent sink.

The [pinned routing graph][ref-route] selects the six largest sigmoid scores and weights the
selected **unbiased** scores by `p_i / max(sum_selected(p), 2^-14)`, scale 1. The real artifact has
no selection bias. An optional supported `exp_probs_b.bias` would change selection only, never the
source of the gathered weights. The shared SwiGLU output is added without a shared sigmoid gate.

Three numerical obligations must stay explicit:

1. Current [common router normalization][route-core] uses `sum + 1e-20`, not the reference's floor.
   This is a semantic difference for tiny scores, not a width-instance problem or a measured GPU
   failure. The existing `Router`/`RouterRule` contract now carries the explicit policy, and the
   reader requests `SelectedSumFloor` for Flint. Selection refuses that policy on every current
   router row; `norm_arg` serves only the two legacy policies. The device floor binding is still
   missing. Add it in the common operation without changing the global divisor, existing model
   arithmetic, or hiding the difference in a Flint-only wrapper.
2. Mainline [rescales Flint's activated expert input before down projection][ref-down] with
   `build_amax_src1_f16`, then restores its scale before router weighting. This protects F16/Q8
   headroom around a maximum magnitude of 1024; it is not an architectural activation clamp.
   Compare the common activation packing at that boundary and large magnitudes. Any needed finite
   range handling belongs to that common operation, with explicit arithmetic/gate evidence.
3. The shared routed owner adds `x + shared` before routed sums; mainline forms
   `(routed + shared) + x`. Accumulation order and activation formats prevent a bitwise-equivalence
   claim from algebra alone. Keep established models unchanged and gate the Flint numerical choice
   against the pinned oracle and the engine's own arithmetic referee.

## 4. Reader and admission: use the existing description

Use `ModelSpec`/`LayerSpec`, `Mixer::Gqa`, `Ffn::Dense`/`Moe`, `Shared`, plain residual and existing
[tensor roles/stems][stems]. No second schema, interpreter, capability catalog or per-token
architecture-name dispatch is needed. At NoPE layers use the existing canonical unused-RoPE
representation (`mode=None`, `dims=0`, `base=0`, no YaRN). Elsewhere use NeoX/128/10000. Preserve
both masks independently even though this artifact's masks coincide.

The reader must classify all 779 descriptors, validate exact ranks/axes and positive checked
geometry, reject unknown/misplaced tensors, and require the untied output. Bias presence becomes
`Router.bias`; it is false here. Read explicit shared width 1024; do not copy `noaux::Experts`, which
requires bias and derives width from a shared-count convention. Invalid mask types/lengths, missing
norms/sinks/shared matrices, an out-of-range block, impossible KV grouping, zero dimensions,
nonfinite epsilon/base, overflow and malformed optional bias all need named refusals.

At this snapshot there is no Flint architecture reader or selected program. A reader-only landing
must keep `program_of=None` and the existing **`Need::Program`** refusal in [coverage][coverage].
Do not claim runtime support by adding the name to an enum or relaxing a shape/type row.

### Concrete common prerequisites

| ID | Current limitation | Required common change before an executable attach |
|---|---|---|
| G1 | NoPE normalization-need generation is present; the executable head128 norm-only binding is missing | Preserve the emitted normalization **or** rotation need; bind head128 normalization with `mode=None` truthfully. No identity-RoPE table. |
| G2 | Existing head128/group8 attention lacks the complete sink/SWA contract; MiMo's window pass is K192/V128 | Complete the described head128 operation in `rope_neox` / `flash_gqa`, including genuine NoPE append, sink merge, window/ring positions, prompt and slot paths. Never pad heads to 192 or copy MiMo's body. |
| G3 | The host description/selection policy is explicit; sigmoid256/top6 and its device floor binding are absent, and higher-level `shared_routed` admission requires bias | Add the real instance/floor operation; bind absent bias through `RouteLayer.bias=None`, which the lower `RouteBlock` already supports. Price/load the BF16 router through its supported F32 widening. |
| G4 | Layer42 shared IQ4_XS has routed/GEMM operations but no ordinary dense site, and `BlockPlan` rejects it | Extend the **common** dense/gated block's real layout, load, launch, scratch and prompt binding. Keep the role `SharedExpert`; do not relabel it as routed or silently requantize it. |
| G5 | The common card wrapper uses Q8Act-only buffers and `LegStacks::of`; Q6_K/Q8_0 down do not both pass that path | Carry selected arms and actual down forms through `CardExperts`, `LegBufs::for_downs`, loader, planner, residency and tier. Existing Q6_K/Q8_0 arithmetic is not the missing piece. |
| G6 | Common tier accumulation supports six slots, but F32 `enqueue_handoff_tier` accepts eight | Add the six-slot F32 handoff instance in the existing owner; preserve the eight-slot entry. A DeepSeek Q8_1 handoff is not an interchangeable input contract. |
| G7 | `place_hybrid` rejects paged experts; common `Hybrid<F>` has `RESIDENT=false` | Lift matching plan/load/host-source and resident-slot plumbing in the existing adapters. Do not remove refusal before the loaded body and seat actually implement the capability. |
| G8 | Common original-template, tokenizer and four-EOG support is present through #677/#678 | Bind the existing chat/tool/stop owners when the executable seat attaches; keep the original template and GLM-format tool parser. |

Attention implementation boundary: the existing head128 ring pass and value-head128 sink merge
are promising ingredients, **not an already callable Flint operation**. Their partial segmentation
must agree: a `SEGMENTS` partition over an effective row count cannot be passed to a wrapper using
unrelated `window_segments(513)`. Bind one correct pass/merge path, not a sinkless merge followed
by another merge. A new NoPE entry is an add-class change, even if its internal helper already
accepts `ROT=0`. See [attention owners][flash] and [head normalization][rope].

### Per-role quant admission

[QROWS][qrows] and [placement kernel predicates][kernels] remain authoritative. All eight routed
role/type combinations below have host kernels and card operation entries at this pin. Eligibility
still means the selected program's gate/up **and** down arms, layout and buffers agree.

| Routed role | Observed type | Card operation / down form | Existing integration limit |
|---|---|---|---|
| Gate/up | IQ4_XS | `Iq4xs` | Common leg operation exists |
| Gate/up | Q5_K | `Q5k` | Operation exists even though the default routed-layout column is empty |
| Gate/up | Q4_K | `Q4k` | Common leg operation exists |
| Gate/up | IQ3_S | `Iq3s` | Decode **and prompt GEMM** exist; an older contrary comment is stale |
| Down | Q5_K | `Q5k` / `Q8Act` | Common leg operation exists |
| Down | IQ4_XS | `Iq4xs` / `Q8Act` | Common leg operation exists |
| Down | Q6_K | `Q6k` / `Q8Act` | G5: selected arms and loader, not absent arithmetic |
| Down | Q8_0 | `Q8_0` / `Q8Blocks32` | G5: actual buffer form and accounting |

Host rows require their declared CPU features and block widths; 2048/512 satisfy the listed block
geometries, but unsupported hardware still refuses. The current leg contract requires gate and up
types to match. If a layer's triple does not satisfy the selected intersection, keep that whole
routed layer on the supported host path and report it; residency must not promote it through a
looser rule. Qwen's wider common-leg admission can inform the lift, **not select a Qwen program**.
Attention Q8_0/Q5_K and shared Q8_0/Q5_K/IQ4_XS must be admitted by their own roles as well.

## 5. Required common-owner table

Every row from [new-model.md](../contrib/new-model.md) has a decision. **Attaches** means a design
commitment with named prerequisites, not an already implemented Flint capability. No row has an
architecture-based reason for a private replacement or omission.

| Row | Decision | Current owner | Attachment obligation |
|---|---|---|---|
| Placement plan | **attaches** | `placement::{plan_routed,plan_routed_reserving}`, `SeqTerms`/`SlotsOf`, `runtime::stores` | Actual per-layer layouts, KV/slots, scratch, residency/tier reserves and supported program rule; G5/G7. No family-owned planner. |
| Kernel table | **attaches** | `gguf::QROWS`, `placement::kernels`, `placement::accepts`, `model::arch::coverage` | Admit only callable intersections; G1–G5. Keep `Need::Program` until the complete body is bound. |
| Card expert leg | **attaches** | `gpu::{card_leg,card_experts,card_slots}`, `gpu_deepseek41::shared_routed::Routed` | `CardFamily` 256/6, width2048/expert512; existing dense lead and ungated shared composition; G3–G5. No private GLM/MiMo FFN copy. |
| Residency | **attaches** | `TierBody`/`TierShell`, `ResidencyParts`, `PlanStacks`/`ResidencyGlue` in `gpu::host` | Return real `Some(ResidencyParts)` for an enabled load, sharing slot map/source over routed blocks `2..49`; same type intersection for placement and promotion. Attach NVMe to host service **and** residency reads. |
| Expert tier | **attaches** | `gpu::host::{leg_tier,tier}`, common `CardLeg`, `shared_routed::join_rows` | G5/G6; price staging and actual down forms, test prompt and decode, include host leftovers. Preserve the existing supported-card-count refusal. |
| Prompt walk | **attaches** | `runtime::{prompt,sched}`, `gpu_deepseek41::prompt_walk::{BatchAttention,Batch}` | Bind stage/attend/FFN to common dense/routed scheduling, checked positions and ring-safe ordering; G1–G5. Existing call-end checkpoint marks do not imply arbitrary checkpoint support. |
| Resident slots | **attaches** | `gpu::model::slots::{Slots,SlotRows}`, `placement::slots`, `app::Session` | Slot-local stores/high-water/captures, model-wide weights/host tier/residency/fault word; price every requested slot and never lower `--parallel` silently; G2/G7. |
| Serve seat | **attaches** | `app::arch::Batch`, shared `serve_seats::{hybrid,tiered,rounds}` | Thin facts/registration after G7/G8; implement actual resident-slot open/step and ring-aware keep. Copying `Hybrid` with `RESIDENT=false` or MiMo's private seat is not completion. |
| Timing | **attaches** | `tools/ref/depth-serve.sh`, existing model profile/corpus and `lease_take` | The admitted HTTP seat, pinned artifacts, prediction card and common lease runner. No Flint-only timing driver or performance claim from this design. |

Current owner anchors: [planner][planner], [slots][slots], [card leg][card-leg], [card wrapper][card-experts],
[shared routed composition][shared-routed], [residency][residency], [tier][tier], [handoff][handoff],
[prompt walk][prompt], [application][app], [shared seat][seat], [timing][timing].
The historical checklist's “each family owns its expert leg” and “MiMo residency returns None”
are not current facts: the leg is common and MiMo delegates residency to its shell.

This follows [#609's composition contract](layer-composition.md): resolve concrete bindings at
load/capture time; preserve fused operations, lifetimes, streams, capture addresses and slot-local
writers. No per-token metadata interpreter or second capability catalog. PR [#666](https://github.com/midagedev/bloomery/pull/666)
was active at inspected head `6e33aee3e0826c2e130b5e72b805f54ab65e67a9`, moving Qwen plan/read/walk
seams into `chain`. Rebase implementation onto the landed ownership boundary; do not introduce a
competing abstraction. That Qwen seam is not proof of Flint or Metal support.

## 6. Placement, slots and retention

Use [common layer stores][kv-layout] and [ring rules][stores], not nominal file size as a fit test.
For per-slot context `C`, resident slots `S`, and f16 K/V, the derived KV row is
`4*(128+128)*2 = 2048` bytes. Today's 512-row append-before-read prompt contract needs
`R(C)=min(C, ceil((513+512-1)/256)*256)=min(C,1024)` rows for each sliding store.
Therefore:

`KV(C,S) = S * 2048 * (14*C + 35*min(C,1024))`.

| Per-slot context | One-slot mixed ring/plane KV bytes | One-slot all-plane KV bytes |
|---:|---:|---:|
| 4,096 | 190,840,832 | 411,041,792 |
| 8,192 | 308,281,344 | 822,083,584 |
| 262,144 | 7,589,593,088 | 26,306,674,688 |

These are **derived store terms**, checked with `runtime::stores::ring_rows`, not measured total
VRAM. Reserving only 513 sliding rows would undercount the current prompt implementation. Add
one stage RoPE table of `512*C` bytes under the current f32/128 convention, **not per slot**.
Sinks are static weights; `KvLayout.sink_row` is an unrelated sparse-indexer dummy row and does not
apply. `SeqTerms::plan_kv` already handles the live sequence's `beside` exclusion; do not charge
all shared scratch once per slot or omit actual slot-local buffers.

The encoded payload separates into **23,049,797,632 routed bytes** and **1,999,420,544 other bytes**.
The file is about **23.334 GiB**, below 24 GiB. It does not alone prove a 3090 fit or non-fit.
BF16 routers alone widen from 1,048,576 to **2,097,152 bytes per routed layer** in the existing
resident format. Add actual card packing, allocation granules, KV, RoPE, scratch, tier/residency
buffers and margins before choosing a split. Host-tier attachment is required by the integration
contract, not established merely by comparing decimal GB with binary GiB.

For layer `l`, derive each expert's encoded bytes from its actual row types:
`512*r_gate(2048) + 512*r_up(2048) + 2048*r_down(512)`, where
`r_t(k)=k/block_t*bytes_t`. Resident bytes come from `CardFormat::buffer_bytes`, not that encoded
sum. Use [common activation/leg terms][kernels]: for prompt capacity `P`, six slots/token, the leg
contains packed input for `(P,2048)`, F32 gate/up output `(6P,512)`, **all actual down forms** for
`(6P,512)`, and F32 down output `(6P,2048)`. Add the attention/QKV, router, dense6144/shared1024,
partial-merge, host/tier exchange and slot allocations according to their actual lifetimes.
The existing Q8Act-only tier helper must not underprice Q8Blocks32. Reject invalid/overflowing
geometry before calling helpers whose products are not themselves checked.

A filled 1024-row ring retains cuts of **512**, but not **513**, at context4096; the shared helper
was exercised at both sides. Retention uses `max(written_high_water,current_pos)`, and cuts do not
lower that high-water mark. Slot park/reset/swap must preserve it with store/capture identity.
The 35 unshadowed window stores are checkpoint state even though Flint has no recurrent layers.
Without integrated saved-window restoration, deeper rewind means named refusal or reset/refeed,
not invented prefix reuse. Ordinary slots do not imply partial speculative keep; preserve
`SlotRows::SETTLES_PARTIAL_KEEP=false` until its stronger restoration contract is implemented.

## 7. Tokenizer, template, tools and stopping

The actual metadata was passed to `Tokenizer::from_keys`; the exact official/GGUF template was
passed to `ChatTemplate::parse`. Current-source host probes produced:

```text
pre-tokenizer "flint" is not supported (...)
chat template: block set is not supported: set tool_preamble
chat template: |replace takes no arguments
```

The last result is an independent minimal render of `{{ 'abc' | replace('a', 'x') }}`; the official
template stops earlier at block capture. Full diagnostic strings are in the evidence JSON.
These are observed current refusals, not a claim that a reader or template fix has landed.

- **Pretokenizer:** extend the [common tokenizer][pretok] with the exact ordered split program:
  number runs of 1–3; the declared CJK/Kana ranges; then Unicode case/mark-aware words,
  contractions, punctuation and whitespace. Use the pinned [mainline regexes][ref-pretok] and
  official tokenizer as separate comparators. No normalizer, no automatic BOS/EOS and no
  `glm4`/Qwen/GPT-4o alias. Cover combining marks and range edges, not just ASCII examples.
- **Jinja:** add block `set` capture and the two-argument `replace` filter to the [existing engine][jinja].
  Macro, namespace, `.items()`, `tojson(ensure_ascii=False)` and string escapes already have common
  owners; keep the 2,328-byte original template. Validate their combination rather than install
  a rewritten Flint renderer.
- **Template behavior:** preserve NUL/think-tag removal and role-marker escaping in non-assistant
  text; tools preamble insertion after initial system messages; developer turns; reasoning
  history; raw string versus JSON nonstring arguments; and `<|observation|><tool_response>`
  results. The generation suffix is `<|assistant|>`, not an open `<think>`.
- **Tools/reasoning:** the existing [GLM key/value grammar][glmxml] matches
  `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`.
  Bind `ToolFormat::GlmXml`, existing `ThinkTags`/`MayOpen`, and the common OpenAI argument-object
  adapter. This is source-derived reuse, not yet exercised Flint streaming/HTTP compatibility.
- **Stops:** choose the official/artifact **four-ID EOG contract**. Current [explicit-stop ingestion][vocab]
  reads `tokenizer.bloomery.eog_token_ids`, not this artifact's `tokenizer.ggml.eog_token_ids`.
  Extend common ingestion/validation for the actual key and preserve range/dedup/type invariants;
  never collapse it to scalar EOS. The pinned mainline's Flint **name-based rule** adds user and
  observation but not developer. Verify its effective loaded stop set rather than claim a
  runtime three-stop result from that branch alone. Record any discrepancy separately from
  teacher-forced logit comparison; do not silently discard developer/150018 to match a heuristic.

## 8. Ordered implementation hand-off and acceptance

This design PR changes no runtime registration, admission row, gate threshold or numerical rule.
The next contributor should implement the following dependency order, keeping each common change
with its real caller and independent evidence:

1. **Reader and host contract.** Add the Flint reader/architecture row, exact roles, both masks,
   explicit widths and optional-bias handling. Correct G1 and the common tokenizer/Jinja/EOG
   gaps. Reader-only support remains `Need::Program`; no advertised runnable seat.
2. **Reference and fixture identity.** Pin mainline build/model/header identities. Use the existing
   fixture machinery with `PerLayer` for both arrays. Preserve both dense lead layers, full/SWA
   dense/MoE combinations and transitions, and every observed `(role,type)` pair—not a convenient
   homogeneous subset. In particular retain layer2 IQ3_S, layer4 Q8_0 down, layer8 Q6_K down and
   layer42 shared IQ4_XS. Resolve artifact/redistribution provenance before publishing fixtures.
3. **Common numerical/operation prerequisites.** Complete G2–G5: head128 norm/NoPE/sinks/SWA;
   unbiased sigmoid256/top6 with explicit normalization policy; shared IQ4_XS; actual card-leg
   arms/down forms and matching checked placement/scratch. No admission before callable bindings.
4. **State and tier integration.** Attach common prompt, host service, residency, NVMe and slots;
   implement G6/G7 together with their accounting and refusals. Preserve capture/stream lifetimes,
   ring high-water retention and the host leftovers of a two-card split.
5. **Body, application and seat.** Bind the existing scheduler/composition and thin family facts;
   enable the program/seat only when coverage and load agree. Complete end-to-end gates below,
   then the existing profile-driven timing runner. Do not call completion after a compiling reader.

### Required evidence before runtime support

| Boundary | Consumer-visible acceptance |
|---|---|
| Reader / coverage | Missing/malformed masks and required tensors, axes/counts, optional-bias errors, unknown names, zero/nonfinite/overflow geometry refuse by name; 49 norm needs, 35 rotary needs, 49 sink needs. Unsupported program/site remains a refusal. |
| Tokenizer / chat | Byte-exact mainline and official-tokenizer comparisons on Unicode/number/tag corpus; original-template comparison against Jinja2 for roles/tools/reasoning/escaping; streamed versus whole GLM calls and malformed/truncated errors; all four EOG IDs, no added BOS/EOS. |
| Attention | Head-local versus flattened RMS discrimination; half-pair RoPE and genuine NoPE; sinks once per reduction, zero versus absent sink; lengths512/513/514 and wrap1024; plane/ring, prompt/decode, eager/replay and multiple-slot isolation. |
| Router / FFN | Top6 IDs/weights, ties, absent bias, optional selection-only bias, sums around `2^-14`, large activated values around reference rescaling; dense/shared/routed output and explicit residual-order comparison. No threshold loosening. |
| Quant / placement | Every observed role/type and mixed triple; gate/up mismatch, Q6_K and Q8_0 down forms, shared IQ4_XS; plan-accepted equals loader-accepted, budget-floor/overflow refusals and host/card/tier conservation. |
| Stores / slots | Context boundaries1/512/513/1023/1024/1025 and slots1/2/3; stage RoPE charged once; requested slot count retained; filled-ring cuts512/513, repeated cuts, divergence, park/reset/swap and high-water preservation. |
| Host / residency / tier | Mixed host/card picks; promotion/demotion constrained by the same capabilities; NVMe fill/reuse/eviction and residency-source lifetime; sidecar refusals; two-card **prompt and decode** with six-slot F32 handoff and stable merge. |
| End to end | Pinned-mainline teacher forcing at dense, first routed, sliding/full transition and final layers; exact real mixed-quant artifact, recorded backend/KV/flash settings; HTTP/direct outputs and checkpoint-resume behavior. A header census is not this gate. |

Pure lifts need the existing canonical/no-JIT move proof plus eager/capture/prompt/slot behavior.
New router/NoPE/handoff entries are **add-class** changes with base-plus-intended-entry scans,
spill/resource evidence and FAIL-first owning gates. Numerical guard/order changes are numerical
changes, not moves. Run the actual `just affected BASE` selection and owning existing shared
routed, Qwen/MiMo attention, residency and tier gates; none replaces a Flint oracle set. GPU and
lease-backed timing evidence require a suitable machine and have **not** been produced here.

Future bookkeeping belongs to the existing owners: `model::arch::registry` and its generated
`tools/arch-registry.tsv`; `model-hf`'s fixture CLI/`FIXTURE_SPECS`; `tools/fixture-dirs.tsv`;
`refset::arch` and its pinned oracle/dump/profile; and `tools/{families,gate-class,owners,gate-paths,gate-batch-resources}.tsv`.
Update PTX/unsafe/resource records only from the later implementation's actual proof, not this design.

## 9. Evidence produced for this design

Executed locally, host-only:

- Bounded HTTP-range read at the pinned GGUF revision through the existing Hub reader and parser;
  actual header hash, full tensor descriptors/types/shapes and encoded-byte accounting.
- `gguf::inventory_of` on that same prefix; independent array extraction and identical tensor-byte
  total. Config comparisons for both masks, EOG IDs and vocabulary; exact official/GGUF template bytes.
- The three observed tokenizer/template/filter refusals in §7.
- `runtime::stores::ring_rows` at the nine contexts in the JSON, and `cut_depth_keeps` on both sides
  of the filled-ring512/513 boundary. The displayed KV formula uses those actual returned row counts.

No permanent test of documentation wording, fabricated model output or sparse zero-weight inference
was added. Throwaway host probes used the public APIs above and were not installed as model support.
The companion JSON is a source/evidence snapshot, **not** a second runtime capability registry.

To reacquire and check the measured header without downloading the full weights, run from repository
root with a scratch directory; this reuses the tree's one Python parser and the pinned JSON identity:

```sh
python3 -B - <<'PY'
import hashlib, importlib.util, json, pathlib, tempfile
root = pathlib.Path.cwd()
e = json.loads((root / 'docs/research/flint-attach-evidence.json').read_text())
spec = importlib.util.spec_from_file_location('hf', root / 'tools/bloomery/hf.py')
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
g = e['gguf']
with tempfile.TemporaryDirectory() as tmp:
    hub, prefix = hf.Hub(tmp), str(pathlib.Path(tmp) / 'header.bin')
    meta, tensors, _ = hf.read_header(
        lambda a, b: hub.range(g['repo'], g['file'], a, b, g['revision']),
        prefix, g['file_bytes_from_hub'], 64 * hf.MIB)
    assert meta is not None
    end = hf.gguf_ranges().header(prefix, size=g['file_bytes_from_hub'])[2]
    assert end == g['header_bytes']
    assert hashlib.sha256(pathlib.Path(prefix).read_bytes()[:end]).hexdigest() == g['header_sha256_measured']
    assert len(tensors) == e['arrays_and_counts']['tensor_count']
    total = sum(hf.gguf_ranges().nbytes(g['file'], n, t, d) for n, t, d, _ in tensors)
    assert total == e['tensor_payload_bytes_derived']
    print(f"header={end} tensors={len(tensors)} encoded_bytes={total}: verified")
PY
```

The Python reader intentionally skips array values. Their recorded values came from
`gguf::inventory_of(...).meta`, not `None` values treated as absent GGUF facts. The recipe above
revalidates the **whole header hash**, including those arrays, without a second parser.

[needs]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/models/src/need.rs#L505-L531
[stems]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/stems.rs#L61-L125
[coverage]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/coverage.rs#L1654-L1770
[route-core]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/route_core.rs#L43-L59
[flash]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/flash_gqa.rs#L5575-L5664
[rope]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/rope_neox.rs#L2059-L2127
[qrows]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gguf/src/qtype.rs#L84-L118
[kernels]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/kernels.rs#L96-L280
[planner]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/placement.rs#L2237-L2288
[slots]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/slots.rs#L39-L147
[card-leg]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/card_leg.rs#L180-L248
[card-experts]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/card_experts.rs#L275-L330
[shared-routed]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-deepseek41/src/shared_routed.rs#L475-L644
[residency]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/served.rs#L28-L61
[tier]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/leg_tier.rs#L342-L473
[handoff]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/handoff.rs#L1181-L1235
[prompt]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-deepseek41/src/prompt_walk.rs#L201-L254
[app]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/app/src/arch/mod.rs#L95-L218
[seat]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-gates/src/bin/shared/serve_seats/hybrid.rs#L41-L99
[timing]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/tools/ref/depth-serve.sh#L465-L482
[kv-layout]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/place_hybrid.rs#L403-L592
[stores]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/runtime/src/stores.rs#L47-L112
[pretok]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/tokenizer/src/pretok.rs#L480-L562
[jinja]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/jinja/src/lib.rs#L480-L548
[glmxml]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/serve/src/glmxml.rs#L378-L445
[vocab]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/tokenizer/src/vocab.rs#L330-L351
[ref-flint]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/models/flint.cpp#L80-L194
[ref-route]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-graph.cpp#L2214-L2345
[ref-down]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-graph.cpp#L2500-L2544
[ref-pretok]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-vocab.cpp#L284-L291
