# ModelSpec design — the types, the family readers and the coverage check

The design memo of round `specdesign` (rebuild wave 2, read-only), kept as it reported, under the
lead's dispositions below. Wave 3's `modelspec` round is specified from it; `docs/rebuild.md` §2-1
principle 3 and §2-2 are its frame.

## Lead's dispositions (2026-09-27)

- **§0.2 item 1 is not a correction.** `docs/rebuild.md` and `modelvocab-report.md` speak of
  GLM-5.3-Flash **UD-Q4_K_XL** (the quant that fits the host tier); this memo read the box's
  **UD-Q2_K_XL**, whose routed stacks are IQ2_XS, IQ3_XXS and IQ4_XS. The lead read the UD-Q4_K_XL
  shard headers (Hugging Face range requests of each shard's first 2 MB, 09-27): routed down is Q5_K
  on layers 3-10, 13-43 and 45 (40 of 43 MoE layers) and Q6_K on 11, 12 and 44; gate and up are Q4_K
  except layer 11's Q5_K. So the plan's routed Q5_K holds for the quant that fits, and Qwen3.6-35B-A3B
  UD-Q4_K_XL carries it too (this memo's header). The format owner (§5) must name it.
- **§0.2 item 2 holds.** The V4.1 Q3_K_M header carries no `nextn.*` or `mtp.*` tensor;
  `docs/rebuild.md` §2-2 is corrected.
- **L4 landed as a refusal, not a cap** (`ef80083`): every entry that advances V4.1 positions
  refuses past 16,384 by name (`Hparams::candidate_free_positions`, `Body::check_defined`); the
  serving plan keeps its 32,768-position caches. Building the candidate mask is a triage item.
- **L1, L2, L3, L5, L6, L7, L8: accepted as recommended.** K1–K6 go to 03's `kernelshape`.
- **§8** is disposed in `docs/plan-triage.md`.

## Round memo

- **Round:** `specdesign` (wave r2). Lead: bloomery-aa.
- **Tree:** read-only on main `a23b0ba`, clean.
- **What I did not do:** no edits in any repository, no build, gate, benchmark or git write.
- **Box use:** READONLY header reads only (§0.1, §9).

### 0. Before the sections

#### 0.1 Decompose on paper, resource timeline

**Decomposition.** Each term and where it comes from:
- **Key names and values** come from the file headers:

  | File | Shards | KV | Tensors |
  |---|---|---|---|
  | V4.1 Q3_K_M | 9 | 68 | 1,046 |
  | Qwen3-30B-A3B | 1 | 45 | 579 |
  | DSpark draft | 1 | 59 | 78 |
  | GLM-5.3-Flash UD-Q2_K_XL | 4 | 72 | 1,412 |
  | Qwen3.6-35B-A3B UD-Q4_K_XL | 1 | 54 | 733 |
  | gpt-oss-120b MXFP4 | 1 | 36 | 687 |

- **Layer kinds and counts** come from tensor presence, cross-checked against the key arrays:
  - V4.1: 40 layers = 2 window-only + 18 at ratio 2 + 20 at ratio 1.
  - GLM: 45 trunk layers (34 KDA + 11 MLA) + 1 MTP layer.
  - Qwen3.6: 40 layers = 30 GDN + 10 GQA. Its 733 tensors = 30·9 + 10·6 + 40·2 + 40·8 + 3 [derived]; this matches the header.
- **Instance keys** come from the Rust consts, each cited as path:line.
- **Formats** come from `CardFormat::of`, `of_routed` and `activation_format`, applied to each header's tensor types.
- **Byte counts** are [derived] and show their factors.

**The one term arithmetic could not settle** was what the GLM and Qwen3.6 GGUFs actually say. I predicted it before the dump:
- GLM: arch `glm5next`, `block_count` 46, a per-layer `head_count_kv` array of 0/1, `kpool` 4, `rope.dimension_count` 0, hc fn Q8_0, and routed experts including Q5_K (taken from rebuild.md:159).
- Qwen3.6: arch `qwen35moe`, `block_count` 40 with no nextn, sections [11,11,10,0], `full_attention_interval` 4.

**Outcome.** Everything held except two things:
- The GLM routed types are IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K and Q3_K. There is no Q5_K.
- Qwen3.6's routed stacks include Q5_K, which I did not predict.

The actions for both outcomes were fixed in advance: the keys go into the reader tables verbatim, and the types become the format items of §5.

**No lease was needed.**
- No run took the lease, so no prediction card applies.
- Box calls this half (all `BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0`):
  - `ls /models`;
  - `ls` of five model directories;
  - one stdlib-python header dump, fed on stdin, reading only the KV and tensor-info tables.
- Box calls earlier in the round: the V4.1, Qwen3 and DSpark header dump, and a read of the exllamav3 sources.

**Resource timeline.** The reader and the coverage check are load-time host work on one thread.
- `Split::open` has already parsed the headers ("opening touches the headers only", gguf/src/lib.rs:490-492).
- The reader adds O(layers × stems) hash-map lookups. The check adds O(layers × needs) set lookups. Together that is microseconds to milliseconds.
- Card SMs, PCIe, NVMe beyond the header pages, and DRAM bandwidth: 0.
- Steady-state term: 0. Nothing runs per step. CED's per-call walk reads the same facts per layer whether they come from `LayerKind` or `LayerSpec`.

So the move needs no timed A/B. Its proof is structural (§6). The flow levers (async, bulk, SIMD, SIMT) have nothing to act on in a one-shot header read.

#### 0.2 Corrections to rebuild.md and modelvocab

1. **The "Q5_K routed (card)" item belongs to Qwen3.6, not GLM.**
   - rebuild.md:159 and modelvocab §5(d) put it on GLM-5.3-Flash.
   - In our GLM file the trunk routed stacks are:
     - IQ2_XS: gate/up on 3-10 and 12-44;
     - IQ3_XXS: gate/up on 11; down on 3-10 and 13-43;
     - IQ4_XS: down on 11, 12 and 44.
   - Q2_K and Q3_K appear only on the nextn layer 45.
   - Q5_K routed stacks are in Qwen3.6 UD-Q4_K_XL: down on 0, 2-33 and 35-37; gate/up on 1.
2. **V4.1's three MTP layers are not in our file.**
   - rebuild.md:156 says they are `Role::Unused` today.
   - The Q3_K_M file has no `nextn.*` or `mtp.*` tensor and no `nextn_predict_layers` key.
   - `compress_ratios` has 43 entries. The 3 entries past `block_count` 40 describe blocks the file does not carry.
   - `roles.rs:39-41` matches nothing in it.
3. **GLM's hc fn is Q8_0 (rebuild.md:129): confirmed.** `hc_{attn,ffn}_fn` are Q8_0 [16384, 24] on layers 0-44.
4. **exllamav3's `deepseek_v4.py` is a V4 reader, not a V4.1 reader.**
   - `_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}` (:19) has no entry for V4.1's ratios 2 and 1.
   - It has no engram and no source sharing.
5. **ik's GLM clamp comment contradicts its kernel.**
   - Comment, `src/llama-hparams.cpp:2423`: "clamped SwiGLU (gate clamped before the SiLU, deepseek4 semantics)".
   - Kernel, `ggml/src/ggml-cuda/unary.cu:80-82`: `g = silu(x); g = min(g, limit)`, which clamps after the SiLU.
   - Our `crates/gpu-deepseek41/src/experts.rs:19-22` documents the kernel's rule.
6. **V4.1's candidate pool is not in the GGUF, not implemented, and not capped.**
   - The HF config has `candidate_source_layer_id 20, candidate_topk_blocks 2048, candidate_block_size 8` (v41-ops-report.md:72-74).
   - The pool is inert up to 2,048 · 8 = 16,384 compressed rows on layer 20's ratio-1 stream, i.e. 16,384 positions [derived; v41-ports-report.md:249-250].
   - llama.cpp caps the context there (`src/models/deepseek41.cpp:32-34`).
   - A grep of `crates/gpu-deepseek41/src`, `crates/model/src/arch/deepseek41` and `generate_ds41` finds no cap.
   - So above 16,384 positions our engine departs from the reference without saying so. This is a lead/user decision outside the move (§7 L4).

#### 0.3 Three reference engines: where they differ, which shape this memo follows

| Question | ik, llama.cpp fork, PR #27754 | exllamav3 (box, @0740edc) | mistral.rs @b0f26d5cd | Followed |
|---|---|---|---|---|
| Where a layer's kind comes from | GGUF tensors plus per-layer key arrays: GLM `head_count_kv` 0/1 (ik llama-hparams.cpp:2477-2479, and every MLA layer runs its own indexer at :2483); qwen35moe `recurrent_layers`, else `full_attention_interval` (qwen35moe.cpp:18-26); V4.1 sources walked from tensors (ik llama-hparams.cpp:2147-2176) | HF config lists: `layer_types`, `mlp_layer_types`, `indexer_types` (glm5_next.py); `compress_ratios` → `_RATIO_TO_TYPE` (deepseek_v4.py:19, 54-59); `layer_types` (qwen3_5.py:28-41) | GGUF metadata plus tensor presence (`dense_layer_indices`, normal_config.rs:2412-2419); qwen35moe requires `full_attention_interval` (:1714-1785) | GGUF, tensor presence first (docs/arch-split.md:38); key arrays are a cross-check, refused on disagreement |
| Qwen3.5/3.6 norm offset | the converter folds +1 into every `norm.weight` except `linear_attn.norm` (conversion/qwen.py:401-402) | `RMSNorm(constant_bias = 1.0)` applied to the raw HF weights (qwen3_5.py:370-392, 458-465) | reads the folded GGUF | the GGUF: plain RMSNorm, no offset |
| Qwen3.5/3.6 GDN head order | the converter reorders V heads from grouped to tiled (conversion/qwen.py:453-460) | HF grouped order | "tiled" for qwen35moe (normal_config.rs:1714-1785) | `KHeadMap::Tiled`, fixed by the arch |
| Qwen3.6 rope | IMROPE; `rope.dimension_sections` required (qwen35moe.cpp:8) | M-RoPE only with vision (qwen3_5.py:491-494) | partial NeoX without sections for text | IMROPE as the file says; whether it equals partial NeoX for text is 03's question (K3) |
| GLM collapse | unweighted mean (ik build_glm5next.cpp:520-545; PR: "unweighted mean, not DeepSeek-V4's learned gated head") | `HyperHead(mean=True)` | no GLM-5 support | `Collapse::Mean` |
| GLM SwiGLU clamp | kernel clamps silu(g) (unary.cu:80-82) | `act_limit` (rule not read) | — | ik's kernel rule (the oracle) |
| GLM required keys | ik: most optional, with defaults. PR: required | asserts n_group 1, sigmoid, noaux_tc, norm_topk, qk_rope 0 | — | the PR's required set (§2c) |
| MTP presence | probe the tensors (ik GLM5NEXT; llama.cpp deepseek4.cpp:20-26) | count = `len(compress_ratios) − num_hidden_layers`, built only if the tensors exist (deepseek_v4.py:97-105) | — | tensors decide; the ratio tail only cross-checks |
| DSpark block width | — | `block_size = dspark_block_size + 1`, "counts the seed position" (deepseek_v4.py:91-96): a convention of exllamav3's generator | — | `dflash.block_size` 5 = the pass width with the seed (`[id_last, mask × (w−1)]`, draft/mod.rs:4). The reference block is `[t, noise×4]` with `dspark_block_size` 5 (v41-ops-report.md:229) |
| Missing-item error | one error at a time | assert on the first | `validate` returns the first missing item (normal_registry.rs:861-919) | ours: every item in one error (roles.rs:5-6, placement.rs:764-769) |

### 1. The Rust types

These refine the draft in rebuild.md:135-137:
- **Sources live inside the latent mixer.** They are `Compress.rows` and `StreamTopK.{keys, list}`, typed as `Source<T>`, instead of a `LayerSpec.sources` block.
  - A GQA or delta-rule layer cannot carry a source, by type.
  - `From(l)` always points below its reader, which is CED fact 1.
  - `LayerSpec::sources()` gives CED its `[rows, keys, list]`.
- **The spec carries no tensor types.** The hc fn format, the embedding-row type and the routed-stack types come from the role table's `ModelTensors` (role and type per tensor). The coverage check joins the spec with the role table. Today's `RowTypes` and `LayerKind.shared_down` stay in `Hparams`.
- **One owner per fact.**
  - Engram sites are `Extra::Engram` on the layers themselves, with no separate site list.
  - The csa/hca segments (`Hparams.csa_ratio`/`hca_ratio`, read only by the meta gates) become a method.
- **Drafts.**
  - `mtp` holds the nextn layers the file carries.
  - A draft *file* reads to a `DraftSpec`, which the session pairs with a target. This is today's shape: `DraftHparams::read` plus `attach_features`.
  - The n-gram lookup (`gpu-gates/src/draft.rs:14`, n = 3, 2, 1 hard-coded) is a session draft source, not a model fact.
- **Counts are `u32`,** matching the files' key type. `Hparams` projections cast to `usize` without loss.

Tags: [V41] [Q3] [GLM] [Q36] are the four targets. [V4] is V4-Flash, which is read and refused today (see the paragraph after the code).

```rust
// crates/models — host-only; depends on `gguf`.
pub type LayerIdx = u32; // trunk index, 0-based

pub struct ModelSpec {
    pub arch: Arch,                  // general.architecture                               [all]
    pub hidden: u32,                 // embedding_length: values per residual row           [all]
    pub vocab: u32,                  // vocab_size, else tokenizer.ggml.tokens' length      [all]
    pub ctx_train: u32,              // context_length: positions                           [all]
    pub rms_eps: f32,                // attention.layer_norm_rms_epsilon, every block norm  [all]
    pub layers: Vec<LayerSpec>,      // the trunk: block_count − nextn layers, in order     [all]
    pub mtp: Vec<LayerSpec>,         // nextn layers the file carries; empty: none          [GLM 1]
    pub hc: Option<HcSpec>,          // None: no layer is Residual::Hc                      [V41, GLM]
    pub engram: Option<EngramSpec>,  // None: no layer carries Extra::Engram                [V41]
    pub chat: ChatSpec,              //                                                     [all]
}
pub enum Arch { Deepseek41, Deepseek4, Qwen3Moe, Qwen35Moe, Glm5Next }

pub struct LayerSpec {
    pub mixer: Mixer,
    pub ffn: Ffn,
    pub residual: Residual,
    pub extras: Vec<Extra>,          // empty except V4.1 layers 1 and 14
}
pub enum Mixer { Gqa(Gqa) /* [Q3, Q36] */, Latent(Latent) /* [V41, GLM] */, DeltaRule(DeltaRule) /* [Q36, GLM] */ }

pub struct Gqa {
    pub heads: u32,        // attention.head_count: query heads                        [Q3 32, Q36 16]
    pub kv_heads: u32,     // attention.head_count_kv; heads % kv_heads == 0           [Q3 4, Q36 2]
    pub head_dim: u32,     // attention.key_length = value_length, values              [Q3 128, Q36 256]
    pub rope: Rope,
    pub qk_norm: bool,     // per-head RMS gain on q and k before the rope             [Q3, Q36: true]
    pub out_gate: bool,    // attn_q also writes a per-head gate; out ×= σ(gate)        [Q36]
}

pub struct Latent {
    pub heads: u32,                  // attention.head_count; one latent row per position [V41 64, GLM 64]
    pub q_lora: u32,                 // attention.q_lora_rank, values                     [1280, 1536]
    pub latent: u32,                 // cached row, values: V41 key_length, GLM kv_lora_rank [512, 512]
    pub up: LatentUp,
    pub rope: Option<Rope>,          // None: NoPE (rope.dimension_count 0)               [V41 Some, GLM None]
    pub q_head_norm: bool,           // per-head RMS norm on the query                    [V4 true; V41, GLM false]
    pub out: LatentOut,
    pub window: Option<u32>,         // raw positions attended; None: every position      [V41 128, GLM None]
    pub sinks: bool,                 // per-head softmax sink, attn_sinks                 [V41]
    pub compress: Option<Compress>,  // None: no compressed stream                        [V41 2-39]
    pub select: Option<Selector>,    // None: no top-k — the whole stream [V4 HCA] or every position
}
pub enum LatentUp { KeqV /* the latent is each head's K and V [V41] */,
                    Absorbed { qk: u32, v: u32 } /* key_length_mla / value_length_mla [GLM 256/256] */ }
pub enum LatentOut { Grouped { groups: u32, rank: u32 } /* output_group_count / output_lora_rank [V41 8/1024] */,
                     Plain /* attn_output, heads·v → hidden [GLM] */ }

pub struct Compress {
    pub ratio: u32,                  // compress_ratios[l]: tokens pooled per row, > 0    [V41 2 on 2-19, 1 on 20-39]
    pub rows: Source<Compressor>,    // who writes the stream this layer reads
}
pub struct Compressor {
    pub gated: bool,                 // attn_compressor_gate present                      [V41 2, 8, 14]
    pub ape: bool,                   // attn_compressor_ape present                       [V4]
    pub overlap: bool,               // attn_compressor_kv is 2·latent wide               [V4]
}
pub enum Source<T> { Own(T) /* this layer writes it */, From(LayerIdx) /* an earlier layer's Own */ }

pub enum Selector {
    StreamTopK {                     // [V41, V4]
        heads: u32,                  // attention.indexer.head_count                     [32]
        d: u32,                      // attention.indexer.key_length, values per key     [128]
        k: u32,                      // attention.indexer.top_k, rows kept per query     [512]
        keys: Source<IndexKeys>,     // who writes the index keys (indexer.attn_k)
        list: Source<()>,            // who scores (indexer.attn_q_b + indexer.proj)
        candidates: Option<Candidates>, // None: every row is a candidate
    },
    TokenPool {                      // [GLM]
        heads: u32,                  // attention.indexer.head_count                     [32]
        d: u32,                      // attention.indexer.key_length                     [128]
        top_k: u32,                  // attention.indexer.top_k, tokens: top_k/pool pools + a pool−1 tail [2048]
        pool: u32,                   // attention.indexer.kpool, tokens per pooled key   [4]
        key_eps: f32,                // attention.layer_norm_epsilon, the key LayerNorm  [1e-6]
    },
}
pub enum IndexKeys { FromRows /* proj of the pre-rope rows, RMS norm, rope tail, Hadamard [V41] */,
                     Compressor(Compressor) /* indexer_compressor_* [V4] */ }
pub struct Candidates {              // [V41] two-level pool; not in the GGUF (§4)
    pub source: LayerIdx,            // whose index scores pick the blocks               [20]
    pub blocks: u32,                 // blocks kept                                       [2048]
    pub block: u32,                  // rows per block                                    [8]
}

pub struct DeltaRule {
    pub kind: DeltaKind,
    pub k_heads: u32,                // Q36 ssm.group_count; GLM head_count               [16; 64]
    pub v_heads: u32,                // Q36 ssm.time_step_rank; GLM head_count            [32; 64]
    pub d: u32,                      // values per k and v head: ssm.state_size / kda.head_dim [128; 128]
    pub conv: u32,                   // ssm.conv_kernel: causal depthwise taps            [4; 4]
}
pub enum DeltaKind {
    Gdn { khead_map: KHeadMap },     // scalar decay per v-head; one conv over q|k|v; out gated by z [Q36]
    Kda { gate_lower_bound: f32 },   // per-channel decay lower·σ(…); a conv per q, k, v; out gated by σ(g) [GLM −5.0]
}
pub enum KHeadMap { Tiled /* v-head j reads k-head j mod k_heads [Q36] */ }

pub struct Rope {
    pub mode: RopeMode,
    pub dims: u32,                   // rope.dimension_count: rotated values              [V41 64, Q3 128, Q36 64]
    pub base: f32,                   // θ base                                            [V41 1e4 / 1.6e5, Q3/Q36 1e7]
    pub yarn: Option<Yarn>,          // None: plain rope                                  [V41 stream layers Some]
}
pub enum RopeMode { NormTail /* GPT-J pairs, last `dims` values [V41] */,
                    Neox /* NeoX halves, first `dims` values [Q3] */,
                    Imrope { sections: [u32; 4] } /* NeoX halves, pair i's position from its section's axis [Q36] */ }
pub struct Yarn { pub factor: f32 /* [16] */, pub orig_ctx: u32 /* positions [65536] */,
                  pub beta_fast: f32 /* [32] */, pub beta_slow: f32 /* [1] */ }

pub enum Ffn {
    Dense { ff: u32, act: Act },     // feed_forward_length, values                       [GLM 0-2: 12288]
    Moe(Moe),                        //                                                   [all]
}
pub struct Moe {
    pub experts: u32,                // expert_count                                      [384, 128, 288, 256]
    pub top_k: u32,                  // expert_used_count                                 [6, 8, 8, 8]
    pub expert_ff: u32,              // expert_feed_forward_length, values                [2304, 768, 2048, 512]
    pub act: Act,                    // the routed experts'
    pub router: Router,
    pub shared: Option<Shared>,      // None: no shared expert                            [Q3]
}
pub struct Router {
    pub score: Score,
    pub bias: bool,                  // exp_probs_b.bias, selection only                  [V41, GLM]
    pub norm: bool,                  // expert_weights_norm: kept weights renormalized    [all four true]
    pub scale: f32,                  // expert_weights_scale, × on the kept weights       [1.5, 1, 2.5, 1]
    pub hash: bool,                  // routed by the token table ffn_gate_tid2eid        [V4 0-2]
}
pub enum Score { SqrtSoftplus /* 4 [V41] */, Softmax /* 1, or the arch constant [Q3, Q36] */, Sigmoid /* 2 [GLM] */ }
pub struct Shared {
    pub ff: u32,                     // values: V41 count × expert_ff; GLM, Q36 expert_shared_feed_forward_length [2304, 2048, 512]
    pub act: Act,
    pub sigmoid_gate: bool,          // out ×= σ(x · ffn_gate_inp_shexp)                  [Q36]
}
pub enum Act {
    SwiGlu { limit: Option<f32> },   // None: silu(g)·u. Some(L): min(silu(g), L)·clamp(u, ±L);
                                     // L ≤ 1e-6 clamps nothing (ik unary.cu:80-82, :315) [V41, GLM 10.0]
}

pub enum Residual { Plain /* [Q3, Q36, GLM mtp] */, Hc /* the owning spec's HcSpec [V41, GLM trunk] */ }
pub struct HcSpec {
    pub streams: u32,                // hyper_connection.count                            [4]
    pub sinkhorn: u32,               // hyper_connection.sinkhorn_iterations              [20]
    pub eps: f32,                    // hyper_connection.epsilon: floor on the mixes      [1e-6]
    pub mix: HcMix,
    pub collapse: Collapse,          // the reader refuses LastMix without Lagged
}
pub enum HcMix { Lagged /* fold by the previous sublayer's pre [V41] */, Own /* [GLM, V4] */ }
pub enum Collapse { LastMix /* the last FFN's unused pre [V41] */, Mean /* [GLM] */, Head /* output_hc_* [V4] */ }

pub enum Extra { Engram }            // [V41 1, 14]
pub struct EngramSpec {
    pub heads: u32,                  // engram.head_count                                 [8]
    pub max_ngram: u32,              // engram.max_ngram_size: n-grams of 2..=max tokens  [4]
    pub key_length: u32,             // engram.key_length, values per gathered row        [256]
}

pub struct ChatSpec {
    pub pre: String,                        // tokenizer.ggml.pre                         [all]
    pub template: Option<String>,           // tokenizer.chat_template; None: the file has none
    pub tools: Option<ToolFormat>,          // None: no parser; a request with tools is refused by name
    pub reasoning: Option<ReasoningFormat>, // None: output not split
}
pub enum ToolFormat { Dsml }                // [V41]
pub enum ReasoningFormat { ThinkSpan }      // [V41] <think>…</think>

pub enum DraftSpec { Block(BlockDraft) }    // a draft file, general.architecture "dflash" [V41's DSpark]
pub struct BlockDraft {
    pub hidden: u32,                 // embedding_length; must equal the target's         [5120]
    pub vocab: u32,                  // must equal the target's                           [129280]
    pub rms_eps: f32,                //                                                   [1e-20]
    pub hc: HcSpec,                  //                                                   [4, 20, 1e-6, Lagged, LastMix]
    pub layers: Vec<LayerSpec>,      // window-only Latent (plain rope at 1e4), Moe 128/3, Hc [3]
    pub width: u32,                  // block_size: positions of one pass, seed included  [5]
    pub target_layers: Vec<LayerIdx>,// the feature is the mean of the hc streams entering each [37, 38, 39]
    pub mask_token: u32,             // tokenizer.ggml.mask_token_id                      [128799]
    pub markov_rank: u32,            // markov_w1's rows — no key carries it              [256]
}
```

Derived as methods, not fields:
- `LayerSpec::sources()`, where `Own` resolves to the layer itself;
- `Gqa::group()`;
- `ModelSpec::{dense_lead, hash_layers, ratio_segments, engram_sites}`, for the meta gates.

**The V4-only fields.** None of the four targets needs these:
- `Compressor.{ape, overlap}`;
- `IndexKeys::Compressor`;
- `Latent.q_head_norm`;
- `Router.hash`;
- `Collapse::Head`;
- `select: None` beside `compress: Some` (the dense stream).

They stay because the deepseek reader reads V4-Flash today, and `crates/model/tests/deepseek4_meta.rs:480-508` pins V4's refusal: ten features from `PlanInputs::unimplemented()` (deepseek41/place.rs:77-124), checked against `tests/common/deepseek4_pins.rs:184-204`.
- With these fields, the coverage check derives that refusal from the spec, and V4 becomes the check's FAIL-first specimen (§6).
- Without them, the reader needs a second output, `unsupported: Vec<Unimplemented>`. That is a second refusal channel next to the coverage check.
- Recommendation: keep the fields. V4-Flash is the planned next model (rebuild.md:310). §7 L3.

**gpt-oss-120b does not fit without additions.** From the header: `gpt-oss`, 36 layers, hidden 2880, GQA 64/8 × 64, `sliding_window` 128, yarn 32/4096 at base 150,000, 128 experts top-4, expert ff 2880.
- It needs five fields:
  - `Gqa.bias`: q/k/v/output `.bias` on every layer.
  - `Gqa.window: Option<u32>`: the alternating pattern is not a key; llama.cpp sets it in code.
  - `Gqa.sinks`: `attn_sinks` [64].
  - `Moe.expert_bias`: `ffn_*_exps.bias` [2880, 128].
  - `Router.logit_bias`: `ffn_gate_inp.bias` [128]. It is added to the logits, unlike `exp_probs_b`, which only steers the choice.
- It needs one new variant: `Act::SwiGluOai { alpha, limit }`. Neither value is a key in the file.
- It fits as is:
  - Its router is `SOFTMAX_WEIGHT` (ik llama-hparams.h:17). A softmax over the k kept experts equals `Score::Softmax` with `norm` [derived]; only the sum order differs.
  - `qk_norm: false`.
  - NeoX rope with `Yarn`.
- The rest are instance and format items: GQA (64, 8), MXFP4 routed stacks, pre `gpt-4o`, and its tool format.

**Qwen3 dense (`qwen3`, modelvocab A:489) fits** Gqa + `Ffn::Dense` + `Residual::Plain`.
- It needs one new field, `head_tied: bool`, for 0.6B, 1.7B and 4B, which have no `output.weight`.
- Its GROUP values 2, 4, 5 and 8 are GQA instances (PACK 2, 4, 1 and 8), not new types.

### 2. Reader tables

Common to every reader:
- Tensor presence decides a layer's kind.
- A key array that disagrees with the tensors is refused by name.
- Every tensor name must have a role (roles.rs:5-6).
- One read is `Result<ModelSpec, PlacementError>`.

#### 2a. `deepseek`: `deepseek41` (then `deepseek4`; `dflash` reads to `DraftSpec`)

Key prefix is `deepseek41.`. "Today" is `crates/model/src/arch/deepseek41/hparams.rs` unless another file is named.

| Key | Field | V4.1 value | Rule | Today |
|---|---|---|---|---|
| general.architecture | arch | deepseek41 | deepseek41 or deepseek4 | arch/mod.rs:32-45 |
| block_count | layers (+ mtp) | 40 | required | :402 |
| embedding_length | hidden | 5120 | required | :457 |
| vocab_size / tokenizer.ggml.tokens | vocab | absent / 129,280 | either | arch/mod.rs:151-166 |
| context_length | ctx_train | 1,048,576 | required | :469 |
| attention.layer_norm_rms_epsilon | rms_eps | 1e-20 (f32 9.9999997e-21) | required | :466 |
| attention.head_count | Latent.heads | 64 | required | :458 |
| attention.head_count_kv | (the latent is shared) | 1 | required | :459 |
| attention.key_length = value_length | Latent.latent, up KeqV | 512 = 512 | required, equal | :403-411 |
| attention.q_lora_rank | Latent.q_lora | 1280 | required | :461 |
| attention.output_group_count / output_lora_rank | LatentOut::Grouped | 8 / 1024 | required | :462-463 |
| rope.dimension_count | Rope.dims | 64 | required, ≤ latent | :412-419 |
| rope.freq_base | Rope.base, window-only layers | 10,000 | required | :552 |
| attention.compress_rope_freq_base | Rope.base, stream layers | 160,000 | required | :561 |
| rope.scaling.{type, factor, original_context_length, yarn_beta_fast, yarn_beta_slow} | Rope.yarn, stream layers | yarn, 16, 65,536, 32, 1 | type must be yarn | :512-570 |
| attention.sliding_window | Latent.window | 128 | required | :465 |
| attention.compress_ratios | Compress.ratio | i32×43: 0,0, 2×18, 1×20, 0,0,0 | required; first 40 read | :1215-1234 |
| attention.indexer.{head_count, key_length, top_k} | StreamTopK | 32, 128, 512 | required | :420-424 |
| hyper_connection.{count, sinkhorn_iterations, epsilon} | HcSpec | 4, 20, 1e-6 | required | :474-476 |
| expert_count / expert_used_count | Moe | 384 / 6 | required | :598-599 |
| expert_gating_func | Router.score | 4 = SqrtSoftplus (ik llama-hparams.h:18) | must be 4 | :589-596 |
| expert_shared_count | Shared (ff = count × expert_ff) | 1 | required | :600 |
| expert_feed_forward_length | Moe.expert_ff | 2304 | required | :601 |
| expert_weights_scale / expert_weights_norm | Router.scale / norm | 1.5 / true | required | :602-603 |
| swiglu_clamp_exp / swiglu_clamp_shexp | Act limits (routed / shared) | f32×40, all 10.0 | required | :429-430 |
| hash_layer_count | Router.hash on the layers below it | 0 | required (llama.cpp: optional, deepseek4.cpp:54-55) | :426 |
| leading_dense_block_count | cross-check of the dense lead | absent | optional, must agree | :1186-1210 |
| engram.{layer_ids, head_count, max_ngram_size, key_length} | Extra::Engram, EngramSpec | [1,14], 8, 4, 256 | required for deepseek41 | :614-655 |
| engram.{multipliers, primes, offsets, token_map, pad_id} | — (hash data, engram crate) | u64×8, u64×48, u64×48, i32×129,280, 2 | required | engram/src/hash.rs:76-157 |
| nextn_predict_layers | mtp count | absent | optional | not read |
| tokenizer.ggml.pre | chat.pre | joyai-llm | must be in `Pre::NAMES` | tokenizer/src/pretok.rs:210-215 |
| tokenizer.chat_template | chat.template | 6,917 chars | optional | serve |

**Tensor presence → kind:**
- `ffn_gate_inp` means Moe; otherwise Dense. Dense layers must lead (:1069, :1186-1210).
- `attn_compressor_kv` means `rows = Own`. `_gate` means gated, `_ape` means ape, and a kv width of 2·latent means overlap. Any other width is refused (compressor_of :814-836).
- `indexer.attn_k` means keys Own. `indexer.attn_q_b` means list Own. `indexer_compressor_*` means `IndexKeys::Compressor` [V4].
- Each source is the last owner at or before the layer (walk_streams :954-1031).
- `ffn_gate_tid2eid` means `Router.hash` (:1103-1130).
- `engram_embd` is present at layer l exactly when l is in `engram.layer_ids` (:1163-1183).
- `output_hc_*` present means `Collapse::Head` [V4]. Absent means Lagged + LastMix (:1136-1159).
- `nextn.*` and `mtp.*` go to `mtp`; today they are `Role::Unused` (roles.rs:39-41).

**Refusals by name that exist today:**
- gating ≠ 4;
- value_length ≠ key_length;
- rope dims larger than the head;
- scaling type other than yarn, or factor ≤ 0;
- the ik walk's four checks: a missing source, a source at another ratio, a pooling compressor without its gate, a third ratio;
- our two walk checks: a compressor at ratio 0, and no compressed layer at all;
- a bad compressor width;
- foreign tensors (:765-788);
- engram key errors;
- a disagreeing leading-dense key;
- tensor names with no role, all listed at once (roles.rs:80-113).

**Where the spec conversion is stricter.** `walk_streams` records `last_key` and `last_idx` before the ratio-0 `continue` (:965-982). So today the reader accepts a window-only layer that carries `indexer.attn_k` or `indexer.attn_q_b`, and CED turns itself off (ced.rs:230-236). The spec cannot hold that case, so the conversion refuses it. No file on the box has it.

**`deepseek4` differs** in four ways: engram keys must be absent, streams are per-layer (own_streams :847-941), `q_head_norm` is true (:467), and the collapse is the trained head.

**`dflash`** reads through `DraftHparams::read` (dspark.rs:110-164):
- the same attention, hc and expert keys;
- `compress_ratios` all 0;
- `hash_layer_count` 0;
- `block_size` 5, `target_layers` [37,38,39], `mask_token_id` 128,799;
- `markov_rank` taken from `markov_w1`'s dims;
- the rope-scaling and indexer keys are present but not read (dspark.rs:16-20).

#### 2b. `qwen`: `qwen3moe` [Q3] and `qwen35moe` [Q36]

Line numbers: qwen3moe/hparams.rs for Q3, llama.cpp qwen35moe.cpp for Q36.

| Key (`<arch>.`) | Field | Qwen3-30B-A3B | Qwen3.6-35B-A3B | Rule |
|---|---|---|---|---|
| block_count | layers (+ mtp) | 48 | 40 | required (:103) |
| embedding_length | hidden | 2048 | 2048 | required (:104) |
| context_length | ctx_train | 262,144 | 262,144 | required (:135) |
| attention.layer_norm_rms_epsilon | rms_eps | 1e-6 | 1e-6 | required (:133) |
| attention.head_count | Gqa.heads | 32 | 16 | required (:105) |
| attention.head_count_kv | Gqa.kv_heads | 4 | 2 | defaults to heads (:107); heads % kv == 0 |
| attention.key_length = value_length | Gqa.head_dim | 128 | 256 | defaults to n_embd/n_head (:118-127) |
| rope.freq_base | Rope.base | 1e7 | 1e7 | required (:182) |
| rope.dimension_count | Rope.dims | absent → 128 | 64 | Q3: must equal the head (:150-157); Q36: partial |
| rope.dimension_sections | Imrope sections | absent | [11,11,10,0] | Q36 required (qwen35moe.cpp:8) |
| rope.scaling.{type, factor} | — | absent | absent | refused unless none / 0 or 1 (:158-179) |
| expert_count / expert_used_count | Moe | 128 / 8 | 256 / 8 | required (:191-199) |
| expert_feed_forward_length | expert_ff | 768 | 512 | required, > 0 (:202) |
| expert_shared_feed_forward_length | Shared.ff | absent | 512 | Q36 |
| expert_shared_count | — | absent | **absent** | Q3 refuses ≠ 0 (:206-211). Q36 must detect the shared expert by its `ffn_*_shexp` tensors |
| expert_gating_func / weights_norm / weights_scale | Router | absent → arch constants Softmax, true, 1 | absent → Softmax, true, 1 (qwen35moe.cpp:499-508) | refused if present and different (:208, :214, :220, :226) |
| feed_forward_length | — | 6144, **unread** | absent | Q3 has no dense layer (:230-242) |
| ssm.conv_kernel / inner_size / state_size / time_step_rank / group_count | DeltaRule conv / (check) / d / v_heads / k_heads | — | 4 / 4096 / 128 / 32 / 16 | required (qwen35moe.cpp:11-15); refuse if inner_size ≠ v_heads × d |
| attention.recurrent_layers | kind cross-check | — | absent | preferred when present (:20) |
| full_attention_interval | kind cross-check | — | 4 | defaults to 4 (:21-25) |
| nextn_predict_layers | mtp | absent | absent | optional |
| general.sampling.{top_k, top_p, temp} | — | absent | 20, 0.95, 1.0 | unread (§7 L6) |
| tokenizer.ggml.pre | chat.pre | qwen2 | qwen35 | |
| tokenizer.ggml.tokens | vocab | 151,936 | 248,320 | |

**Tensor presence → kind** (Qwen3.6, names from its header):
- GDN, layers 0-2, 4-6, …, 36-38: `attn_qkv` [2048, 8192], `attn_gate`, `ssm_conv1d` [4, 8192], `ssm_dt.bias`, `ssm_a`, `ssm_alpha`, `ssm_beta` [2048, 32], `ssm_norm` [128], `ssm_out`.
- GQA, layers 3, 7, …, 39: `attn_q` [2048, 8192], `attn_k`/`attn_v` [2048, 512], `attn_q_norm`/`attn_k_norm` [256], `attn_output`. `attn_q` rows = 16 × 256 × 2, so rows = 2 × heads × head_dim means `out_gate`.
- `ffn_gate_inp_shexp` [2048] means `Shared.sigmoid_gate`.
- `post_attention_norm` is the pre-FFN norm (Qwen3 calls it `ffn_norm`). That is a role-table row.
- Qwen3: every layer carries the twelve stems of names.rs:85-98. `attn_q_norm`/`attn_k_norm` mean `qk_norm` (qwen3moe/roles.rs:15-16).

**New refusals:**
- the kind from the tensors differs from `recurrent_layers` or the interval;
- `inner_size` ≠ v_heads × d;
- v_heads % k_heads ≠ 0;
- qwen35moe without `rope.dimension_sections`;
- an MTP layer with its own `nextn.embed_tokens` or `nextn.shared_head_head`, refused until a model needs it.

#### 2c. `glm`: `glm5next` [GLM]

Values are from the unsloth UD-Q2_K_XL header. Rules are PR #27754's `load_arch_hparams` (required unless marked), with ik llama-hparams.cpp:2376-2485 where it differs.

| Key (`glm5next.`) | Field | Value | Rule |
|---|---|---|---|
| block_count | layers + mtp | 46 | required |
| nextn_predict_layers | mtp count | 1 | optional; < block_count; tensors decide |
| embedding_length / vocab_size / context_length | hidden / vocab / ctx_train | 4096 / 154,880 / 1,048,576 | |
| attention.layer_norm_rms_epsilon | rms_eps | 1e-5 (f32 9.9999997e-6) | required |
| attention.layer_norm_epsilon | TokenPool.key_eps | 1e-6 | PR: required, warns outside (0, 2e-6]. ik: falls back to the RMS eps |
| attention.head_count | Latent.heads; KDA heads | 64 | required |
| attention.head_count_kv (i32×46) | kind: 0 → DeltaRule, 1 → Latent | 1 at 3, 7, …, 43 and 45 | required array; 0 < recurrent < trunk |
| attention.q_lora_rank / kv_lora_rank | Latent.q_lora / latent | 1536 / 512 | required; q_lora > 0 |
| attention.key_length / value_length | check: latent + rope dims | 512 / 512 | refuse on mismatch |
| attention.key_length_mla / value_length_mla | Absorbed{qk, v} | 256 / 256 | required |
| rope.dimension_count | Latent.rope = None | 0 | must be 0 |
| ssm.conv_kernel | DeltaRule.conv | 4 | required, > 1 |
| kda.head_dim | DeltaRule.d | 128 | PR: required. ik: falls back to ssm.state_size |
| kda.gate_lower_bound | Kda.gate_lower_bound | −5.0 | PR: required, < 0. ik: default −5 |
| ssm.group_count | k_heads | absent → 64 | ik: head_count |
| attention.indexer.{head_count, key_length, top_k, kpool} | TokenPool | 32, 128, 2048, 4 | required; top_k % kpool == 0 |
| hyper_connection.{count, sinkhorn_iterations, epsilon} | HcSpec (Own, Mean) | 4, 20, 1e-6 | required (ik: epsilon optional) |
| expert_count / expert_used_count | Moe | 288 / 8 | required |
| expert_group_count / expert_group_used_count | — | 1 / 1 | the PR does not read them; refuse ≠ 1 |
| expert_gating_func | Score::Sigmoid (2, ik llama-hparams.h:16) | 2 | PR: required. ik: default sigmoid |
| expert_feed_forward_length / expert_shared_feed_forward_length | expert_ff / Shared.ff | 2048 / 2048 | required / optional |
| expert_shared_count | Shared present | 1 | PR: required |
| leading_dense_block_count / feed_forward_length | Dense on 0-2, ff | 3 / 12,288 | PR: required; tensors must agree |
| expert_weights_scale / expert_weights_norm | Router.scale / norm | 2.5 / true | PR: required |
| swiglu_clamp_exp / swiglu_clamp_shexp | Act limits: routed / shared and dense | f32×46, all 10.0 | optional (ik: the dense layers take the shared limit, llama-build-context.cpp:1269) |
| general.sampling.{top_p, temp} | — | 0.95, 1.0 | unread |
| tokenizer.ggml.pre | chat.pre | glm4 | |

**Tensor presence, from the header:**
- KDA layers (0-2, 4-6, …, 40-42, 44): `attn_{q,k,v}` [4096, 8192], `ssm_conv1d_{q,k,v}` [4, 1, 8192], `ssm_{f_a,f_b,g_a,g_b}`, `ssm_beta` [4096, 64], `ssm_a` [64], `ssm_dt.bias` [8192], `ssm_norm` [128], `attn_output`.
- MLA layers (3, 7, …, 43 and 45):
  - `attn_q_a` [4096, 1536] with its norm, `attn_q_b` [1536, 16384];
  - `attn_kv_a_mqa` [4096, 512] with its norm;
  - `attn_k_b` [256, 512, 64], `attn_v_b` [512, 256, 64], `attn_output`;
  - the indexer set: `indexer.attn_k` [4096, 128], `indexer.attn_q_b` [1536, 4096], `indexer.k_norm.{weight,bias}`, `indexer.proj` [4096, 32], `indexer_compressor_gate` [4096, 128], `indexer_compressor_ape` [128, 4].
- Layers 0-2 are dense (`ffn_{gate,up,down}`). Layers 3-45 are MoE with `exp_probs_b.bias`.
- The trunk (0-44) carries `hc_*`.
- Layer 45 carries `nextn.{eh_proj, enorm, hnorm, shared_head_norm}` and no hc. The PR's own line, from `gh api repos/ggml-org/llama.cpp/pulls/27754/files` (src/models/glm5next.cpp): "// NextN draft head: an ordinary DSA layer, but a plain residual and no hc_* tensors".
- Shard 1 of this file holds 0 tensors. `Split::open` sums the per-shard counts, so it handles this, but V4.1 (37 tensors in shard 1) never exercised the case.

**Refusals:**
- the kind from the tensors differs from `head_count_kv`;
- an MLA layer without the indexer set (ik would run dense MLA without saying so);
- an hc tensor on the nextn layer;
- group count ≠ 1;
- top_k % kpool ≠ 0;
- rope dims ≠ 0;
- gate lower bound ≥ 0.

**Reader sizes.**
- The qwen3moe reader is 280 lines before its tests (qwen3moe/hparams.rs:1-280).
- A glm reader built from this table would be about the same [estimate].
- The deepseek reader is 1,259 lines (deepseek41/hparams.rs:1-1259) because it carries ik's walk checks and V4 (§7 L2).

### 3. Today's values, field by field (the move proof's table)

[const] marks a fact today's engine takes from a Rust constant. Those are the ones the coverage check must compare against the file.

| Field | V4.1 source | V4.1 value | Qwen3 source | Qwen3 value |
|---|---|---|---|---|
| arch | arch/mod.rs:32-45 | Deepseek41 | arch/mod.rs:69-76 | Qwen3Moe |
| hidden | hparams.rs:457 | 5120 [const] ROW 5120 (engram_gate.rs:57); checked chain/glue.rs:611 | qwen3moe/hparams.rs:104 | 2048 [const] NORM_K 2048 (router.rs:135; k ≤ NORM_K at :1358); multiple of 256 (body.rs:146-153) |
| vocab | arch/mod.rs:151-166 | 129,280 | same | 151,936 (an argument; head_argmax.rs has no vocab const) |
| ctx_train | :469 | 1,048,576 | :135 | 262,144 |
| rms_eps | :466 | 1e-20 | :133 | 1e-6 |
| layers.len() | :402 | 40 | :103 | 48 |
| mtp | roles.rs:39-41 | [] | — | [] |
| hc.{streams, sinkhorn, eps} | :474-476 | 4, 20, 1e-6 [const] HC_STREAMS 4, HC_MIX 24 (hc.rs:66, 68); checked chain/attn.rs:811, chain/ffn.rs:849, chain/glue.rs:611, body.rs:2346, draft/block.rs:188 | — | None |
| hc.{mix, collapse} | :1136-1159 | Lagged, LastMix [const] fused into `ds41_hc_post` (hc.rs:38-44) | — | — |
| engram | :437-443, :614-655 | heads 8, max_ngram 4, key_length 256; sites 1, 14 | — | None |
| chat.pre | pretok.rs:210-215 | joyai-llm | same | qwen2 |
| chat.tools / reasoning | api.rs:1550-1555, reasoning.rs:32 | Dsml / ThinkSpan, for any template | same | Dsml / ThinkSpan today, which is wrong (§8) |
| Latent.{heads, q_lora} | :458, :461 | 64, 1280 | | |
| Latent.latent, up | :403-411 | 512, KeqV [const] LATENT 512 (attn.rs:70), WIDTH 512 (compress.rs:62); checked chain/attn.rs:808-809, attn.rs:951-955, 1024-1030 | | |
| Latent.rope | :412-419, :512-570, :1081-1085 | window-only: NormTail 64, base 1e4, no yarn; stream: NormTail 64, base 1.6e5, Yarn{16, 65,536, 32, 1}; dims checked chain/attn.rs:801 | | |
| Latent.q_head_norm | :467 (model == Deepseek4), an arch fact | false | | |
| Latent.out | :462-463 | Grouped{8, 1024}; checked chain/attn.rs:812-813 | | |
| Latent.window | :465 | 128; ring = min(ctx_max, window) (body.rs:2379) | | |
| Latent.sinks | roles (`attn_sinks`) | true on 0-39 | | |
| Compress.{ratio, rows} | :428, :1215-1234, :954-1031 | 2 on 2-19, 1 on 20-39; Own at 2, 8, 14, 20 | | |
| Compressor | :814-836 | gated at 2, 8, 14; no ape; no overlap | | |
| StreamTopK.{heads, d, k} | :420-424 | 32, 128, 512 [const] HEADS 32, HEAD_DIM 128 (indexer.rs:65, 67), key WIDTH 128 (index_key.rs:40); checked indexer.rs:1006, chain/attn.rs:810 | | |
| StreamTopK.keys / list | :954-1031 | keys Own at 2, 8, 14, 20; list Own at 2, 8, 14, 20, 24, 28, 32, 36 | | |
| StreamTopK.candidates | not read | should be Some{20, 2048, 8} (§4) | | |
| Gqa.{heads, kv_heads} | | | :105, :107 | 32, 4 [const] GROUP 8 (flash_gqa.rs:69; prefill asserts it, flash_gqa_prefill.rs:178); pins() body.rs:123-127 |
| Gqa.head_dim | | | :118-131 | 128 [const] HEAD 128 (flash_gqa.rs:67, rope_neox.rs:46); pins() :122 |
| Gqa.rope | | | :149-186 | Neox [const, :181], 128 (key absent → head, :150), base 1e7, no yarn; pins() :130 |
| Gqa.{qk_norm, out_gate} | | | roles.rs:15-16 | true, false |
| Moe.{experts, top_k} | :598-599 | 384, 6 [const] router.rs:60, 63; checked chain/ffn.rs:841, router.rs:717, 780 | :191-192 | 128, 8 [const] router.rs:72, 75; pins() :128-129 |
| Moe.expert_ff | :601 | 2304; ff % 256 checked chain/ffn.rs:849 | :202 | 768; pins() :146-153 |
| Router.score | :589-596 | SqrtSoftplus (IK_SQRT_SOFTPLUS = 4, :31) | :212-217 | Softmax: arch constant; key absent in the file |
| Router.bias | roles (`exp_probs_b`) | true | — | false |
| Router.norm | :603 | true | :218-223 | true: arch constant, key absent; pins() :140-145 requires it |
| Router.scale | :602 | 1.5 | :224-229 | 1: arch constant, key absent |
| Router.hash | :426, :1103-1130 | false (hash_layer_count 0) | — | false |
| Moe.act | :429 | SwiGlu{Some(10)} | — | SwiGlu{None} |
| Shared | :600 + roles | Some{2304, SwiGlu{Some(10)}, no gate} | :206-211 | None |
| Residual | hc tensors on every layer | Hc | | Plain |
| Extra::Engram | :1163-1183 | layers 1, 14 | | — |
| DraftSpec (DSpark) | dspark.rs:110-164 | width 5 [const] MAX_WIDTH (draft/block.rs:73); target_layers [37,38,39]; mask 128,799; markov_rank 256 [const] MARKOV_RANK (markov.rs:41); experts 128/3 [const] experts_mxfp4.rs:63, 66, checked draft/load.rs:274-280 | | |

The third file the proof covers is V4-Flash (`gate-deepseek4-meta`, justfile:1162-1163). Its values are pinned in `tests/common/deepseek4_pins.rs`.

### 4. V4.1's structure as `LayerSpec` data

**Shared by every layer:**
- mixer `Latent{64 heads, q_lora 1280, latent 512, KeqV, q_head_norm false, Grouped{8, 1024}, window 128, sinks}`;
- ffn `Moe{384, 6, 2304, SwiGlu{10}, Router{SqrtSoftplus, bias, norm, ×1.5, no hash}, Shared{2304, SwiGlu{10}}}`;
- residual `Hc`.

**Per layer:**

| Layers | Rope | Compress (ratio, rows) | Select (keys, list) | Extras |
|---|---|---|---|---|
| 0 | window: base 1e4, no yarn | None | None | |
| 1 | window | None | None | Engram |
| 2 | stream: base 1.6e5, yarn | 2, Own{gated} | Own, Own | |
| 3-7 | stream | 2, From 2 | From 2, From 2 | |
| 8 | stream | 2, Own{gated} | Own, Own | |
| 9-13 | stream | 2, From 8 | From 8, From 8 | |
| 14 | stream | 2, Own{gated} | Own, Own | Engram |
| 15-19 | stream | 2, From 14 | From 14, From 14 | |
| 20 | stream | 1, Own{ungated} | Own, Own | |
| 21-23 | stream | 1, From 20 | From 20, From 20 | |
| 24, 28, 32, 36 | stream | 1, From 20 | From 20, Own | |
| 25-27, 29-31, 33-35, 37-39 | stream | 1, From 20 | From 20, From 24 / 28 / 32 / 36 | |

This matches the header:
- compressors on 2, 8, 14, 20;
- gates on 2, 8, 14;
- `indexer.attn_k` on 2, 8, 14, 20;
- `indexer.attn_q_b` on 2, 8, 14, 20, 24, 28, 32, 36.

It also matches the reader's own `served()` test (hparams.rs:1270-1282).

| Item | Field | Schedule derived from it |
|---|---|---|
| CED | `Source::Own` of rows and keys, `sources()`, window, the draft's tap set | `Ced::new` (ced.rs:224-246) sets `every[l]` = owns rows or owns keys; the walk from the top layer down sets each layer's `part`/`full` (ced.rs:256-305). **Fact 1** (every source at or below its reader) holds by type. **Fact 2** (keys only on a compressor's layer) stays a schedule precondition, `CedState::Off(why)` (ced.rs:180-190, 230-236), not a reader refusal; keep that split in the move. `CedLayer::of` (ced.rs:166-174) reads `LayerSpec` |
| `kv_source_layer_ids` [2,8,14,20] (HF only) | `Compress.rows` | stream buffers only on owners (`LayerKv.rows`, body.rs:162); readers attend the owner's rows; the owner writes its latent part at every position (ced.rs:19-22) |
| `index_source_layer_ids` [2,8,…,36] (HF only) | `StreamTopK.list` (`.keys` on 2, 8, 14, 20) | the list is computed on its owner and read at the same position (ced.rs:26-28); key buffers only on key owners |
| `compress_ratios` | `Compress.ratio` | ⌈ctx_max / ratio⌉ rows per owner (body.rs:13); compressor state above ratio 1; KV bytes (kv.rs:68-105) |
| window | `Latent.window` | ring = min(ctx_max, 128) (body.rs:2379); CED's `slots` |
| indexer | `StreamTopK{32, 128, 512}` | scored on list owners (histogram exact top-k) |
| candidate blocks | `StreamTopK.candidates` | nothing today (§0.2 item 6) |
| engram | `Extra::Engram` + `EngramSpec` | the host gathers the rows (StepRows); the MoE sub-layer before the site runs them in its shadow (glue.rs:17-20); the FFN before a site ends with HC_POST alone, then the gate plus the lagged fold (glue.rs:21-24). This is derived from `Engram` combined with `HcMix::Lagged` |
| mHC | `HcSpec{4, 20, 1e-6, Lagged, LastMix}` | HC_PRE per sublayer; HC_POST fused with the next sublayer's fold (hc.rs:38-44); the head reads the last FFN's fold (glue.rs:25-27) |
| DSpark target layers | `BlockDraft.target_layers` | the tap is the mean of the streams entering layers 37, 38, 39 (draft/mod.rs:12-15; `attach_features`, body.rs:141-160); CED widens layers 36-38 to the last `window` positions (ced.rs:11-14; test ced.rs:433-452) |
| unused nextn layers | `ModelSpec.mtp` (empty) | none. A file that carries them would load them into `mtp`, print them on the load line, and not run them |

**V4.1 facts that are not in the file:**
1. The candidate pool.
2. `q_head_norm` = false (hparams.rs:467).
3. The lagged mix and the LastMix collapse. They are inferred from the absence of `output_hc_*` plus the architecture (hparams.rs:1136-1159; llama.cpp deepseek41.cpp:17-23).
4. The rope rule: a NORM-mode tail; window-only layers use plain rope at `rope.freq_base`, stream layers use YaRN at `compress_rope_freq_base` (hparams.rs:512-570).
5. The DSpark tap reads a target layer's input (draft/mod.rs:12-15). The key only names the layers; plan-triage.md:203 records a converter that shifts this by one.
6. The kernel sizes (the [const] rows of §3).
7. The engram hash-key prefix `deepseek41.` (engram/src/hash.rs:41).
8. `exp_probs_b_vl` is marked `Unread` by name (roles.rs:50).

### 5. The coverage check

**Shape.**
- `needs(&ModelSpec, &ModelTensors) -> Vec<(Need, Option<LayerIdx>)>` runs over `layers` only. `mtp` is reported as carried and not run.
- The available set is the union of: the instance table, the format owner, the program set, `Pre::NAMES` and serve's parsers.
- The error is today's `PlacementError::Unimplemented(Vec<Unimplemented>)` (placement.rs:764-769): "N feature(s) of this file are not implemented: <feature> (layers a, b); …".
  - Items are grouped by `unimplemented_list` (placement.rs:87-111) and ordered layer by layer, then model-wide (place.rs:74-76).
- `Need`'s Display keeps today's V4 strings verbatim, so `deepseek4_pins::UNIMPLEMENTED` still holds.

**Instance keys.** PACK is the largest power of two that divides GROUP.

| Op kind | Instance key | Today's instances |
|---|---|---|
| route | (score, E, K) + bias, hash | (SqrtSoftplus, 384, 6) gpu-deepseek41/src/router.rs:60, 63; its shape asserts (:81, :88, :90) also accept 288. (Softmax, 128, 8) arch/qwen3moe/router.rs:72, 75, with PER_LANE == 4 (:142) and hidden ≤ NORM_K 2048 (:135-140). (SqrtSoftplus, 128, 3) fused into experts_mxfp4.rs:63, 66. (Softmax, 64, 6) gpu/src/router.rs:38, 41 (V2-Lite). Sigmoid is compiled but routes nothing (route_core.rs:14-16) |
| gqa_flash | (HEAD, PACK) | (128, 8): flash_gqa.rs:67, 69; prefill asserts GROUP == 8 (flash_gqa_prefill.rs:178) |
| qk norm + rope + append | (HEAD, mode, dims) | (128, Neox, 128): rope_neox.rs:46 |
| latent_attn | (LATENT, ROPE) + up | (512, 64, KeqV): attn.rs:70 |
| compress | (WIDTH, gated, ape, overlap) | (512, gated or ungated, no, no): compress.rs:62 |
| index_score | (HEADS, D, Pool) | (32, 128, stream): indexer.rs:65, 67; keys index_key.rs:40 |
| delta_rule | (D, KDA) | none |
| hc_pre | (STREAMS, fn format) | (4, Q3_K fused): hc.rs:66-71; `check_params` wants Q3_K words (`words = 110 * (k / 256) / 4`, hc.rs:1527). (4, F32): hc_f32.rs |
| hc wiring | (mix, collapse) | (Lagged, LastMix), which is a program property today (hc.rs:38-44) |
| engram | ROW (= hidden) | 5120: engram_gate.rs:57 |
| draft block | MAX_WIDTH, markov rank | 5 (draft/block.rs:73), 256 (markov.rs:41) |
| program (wave 3) | the LayerSpec shapes a family program accepts | deepseek41 chain: Latent KeqV + Compress/StreamTopK + Moe SqrtSoftplus + Hc Lagged + Engram. qwen3moe body: Gqa (no gate, full NeoX) + Moe Softmax without a shared expert + Plain |

**Format owner.** One table per `GgmlType`. `CardFormat::of` (placement.rs:270-301), `of_routed` (:366-368) and `activation_format` (gguf/src/quant.rs:1120-1150) become views of it. Note that `CardFormat::of`'s `None` "is not a refusal by itself: a routed stack of such a type stays on the host" (placement.rs:263-267). The V4.1 family's refusal comes from `unimplemented()` (place.rs:109-113).

| Type | Card, dense | Card, experts | Card kernel exists but placement refuses it | Host qdot activation |
|---|---|---|---|---|
| Q3_K / Q4_K / Q6_K | KQuant | yes | | Q8K / Q8_2X4 / Q8_2X4 |
| Q5_K | KQuant | no (only a dense gemv reads it) | | Q8_2X4 |
| Q5_0, Q5_1 | yes | yes | | Q8_2X4 |
| Q8_0 | Q8_0Planes | yes | | none |
| F32 / BF16 | F32 / Bf16AsF32 | yes | | plain / none |
| MXFP4 | no | no | mxfp4.rs:1-3, experts_mxfp4.rs | Q8_2X4 |
| IQ2_XS, IQ4_XS, Q2_K | no | no | iq.rs:1-4 | none |
| IQ3_XXS | no | no | iq.rs:1-4 | Q8K (quant.rs:1122) |

**Program pins on top of that table (wave 3):**
- The qwen3moe program's `kq_site` calls (body.rs:243-249) take Q4_K only for q, k, output, gate and up, and Q4_K or Q6_K for v and down. `token_embd` and `output` are not pinned there.
- V4.1's chain decodes embedding rows only as bf16 or Q3_K (draft/block.rs:5-8).

**Tokenizer and chat.**
- `Pre::NAMES` (pretok.rs:210-215): deepseek-v3, hunyuan-dense, joyai-llm, qwen2.
- Tools: DSML for every request that has tools (api.rs:1550-1555).
- Reasoning defaults to Deepseek (reasoning.rs:32).

**What it would print.** Today both new files stop at the architecture name:
- `ModelError::UnknownArchitecture("glm5next")` or `("qwen35moe")`, from `Arch::from_name` (arch/mod.rs:69-76), via `AnyEngine::open` (engine.rs:44);
- or, on the V4.1 binaries, `metadata general.architecture: is "glm5next"; the deepseek41 module reads deepseek41 and deepseek4` (:32-45).

The lists below are [derived]: what a reader plus the coverage check would print, computed from the real headers against today's tables.

**GLM-5.3-Flash UD-Q2_K_XL, over the trunk (0-44).** Per layer:
1. delta rule KDA: d 128, 64 heads, conv 4.
   Layers 0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, 44 (34 layers).
2. latent attention 512 × rope 0, absorbed qk 256 / v 256.
   Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 (11 layers).
3. token-pool indexer: 32 × 128, pool 4, top 2048 plus a tail of 3, LayerNorm keys with bias.
   Same 11 layers.
4. router: sigmoid, 288 experts, top 8, with a selection bias.
   Layers 3-44.
5. dense SwiGLU layer: ff 12,288, limit 10.
   Layers 0, 1, 2. `dense_lead` is read (hparams.rs:1186-1210), but no GPU code reads it.
6. hc_pre with a Q8_0 fn (hc.rs:1527 wants Q3_K words).
   Layers 0-44.
7. hc mix folded by the sublayer's own `pre`.
   Layers 0-44.
8. Routed stacks, by format owner:
   - IQ2_XS (gate/up 3-10, 12-44) and IQ4_XS (down 11, 12, 44): no card format and no host format, so neither tier runs them.
   - IQ3_XXS (gate/up 11; down 3-10, 13-43): no card format; host format Q8K.

Model-wide:

9. hc collapse by an unweighted mean. `ds41_hc_mean` (hc.rs:48-50) is a candidate kernel.
10. A recurrent-state slot per sequence [derived, assuming f32 state]:
    - KDA state: 34 layers × 64 heads × 128 × 128 × 4 B = 142,606,336 B;
    - conv state: 34 layers × 3 convs × (4 − 1) kept inputs × 8,192 channels × 4 B = 10,027,008 B.
11. `token_embd` rows in Q5_K.
12. Pre-tokenizer `glm4` (llama.cpp maps it to CHATGLM4 with no BOS, llama-vocab.cpp:2269-2272).
13. A tool-call parser for this template (10,648 chars, starting `[gMASK]<sop>`).

Load line, not a refusal: MTP layer 45 is carried and not run. Its gate/up are Q2_K and its down is Q3_K.

Not items, because instances already exist:
- the SwiGLU limit (experts.rs:19-22);
- 4 streams, mix 24, Sinkhorn 20;
- Q8_0, Q5_K and Q6_K dense matrices;
- the Q4_K output.

**Qwen3.6-35B-A3B UD-Q4_K_XL.** Per layer:
1. delta rule GDN: d 128, 16 k-heads and 32 v-heads (tiled), conv 4.
   Layers 0-2, 4-6, …, 36-38 (30 layers).
2. GQA flash, head 256, pack 8.
   Layers 3, 7, …, 39 (10 layers).
3. per-head QK norm plus rope: head 256, IMROPE [11,11,10,0], 64 of 256 dims.
   Same 10 layers. `pins()` refuses rope dims ≠ head (body.rs:130).
4. attention output gate: sigmoid, interleaved with q.
   Same 10 layers.
5. router: softmax, 256 experts, top 8.
   Layers 0-39.
6. shared expert, ff 512, with a sigmoid gate.
   Layers 0-39.
7. Routed Q5_K stacks (down on 0, 2-33, 35-37; gate/up on 1): `of_routed` keeps them off the card, and the qwen program is whole-card (engine.rs:36). Nothing runs them.
8. Q8_0 weights in attention, GDN and the shared expert: they fail the program's per-tensor pins (body.rs:243-249). Whether the Q8_0 `token_embd` and `output` fit their paths is unverified.

Model-wide:

9. A recurrent-state slot [derived, assuming f32 state]:
   - GDN state: 30 layers × 32 v-heads × 128 × 128 × 4 B = 62,914,560 B;
   - conv state: 30 layers × (4 − 1) kept inputs × 8,192 channels × 4 B = 2,949,120 B.
10. A program that runs GDN and GQA layers in one trunk.
11. Pre-tokenizer `qwen35`, which is its own pre type (llama-vocab.cpp:2249-2251).
12. A tool-call parser for this template (8,057 chars).

Not items:
- rms eps;
- vocab 248,320 (an argument);
- Q4_K and Q6_K routed stacks;
- the softmax + norm router semantics.

### 6. Wave-3 migration (move class)

**Steps:**
1. **Types.** Create `crates/models` with §1's types. `Role` and `Unimplemented` move there and are re-exported from `model::placement`, which is a rename.
2. **Readers wrap.**
   - `deepseek::read` and `qwen::read` are `Hparams::read` plus `roles::classify` plus a pure `spec_of(&Hparams, &ModelTensors)`.
   - `DraftSpec::read` wraps `DraftHparams::read`.
   - There is no second key reader. The readers live at `crates/model/src/arch/<family>/spec.rs` and move into `crates/models` in wave 4.
3. **Coverage check.** `needs()` plus a hand-written available table: the rows of §5, each with its path:line. opslib's table replaces it when that lands.
4. **Call sites that read the spec first:**
   - `PlanInputs::read` (deepseek41/place.rs:55-63): `unimplemented()` (:77-124) becomes the coverage check, with the same error and the same strings.
   - qwen3moe `PlanInputs::read` (qwen3moe/place.rs:72-77): the check is added; it finds nothing today.
   - `Ced::new` and `CedLayer::of` (ced.rs:166-174, :230) take `&[LayerSpec]`.
   - `AnyEngine::open` (engine.rs:39-58) dispatches on `ModelSpec.arch`.
   - `attach_features` (shared/ds41_dspark.rs:107) takes `BlockDraft.target_layers`.
   - The load lines print what is carried but not run.
5. **Untouched until wave 4:**
   - the layer programs: `gpu-deepseek41/src/{chain/*, body.rs, body/prefill.rs, draft/*}` and `gpu/src/arch/qwen3moe/*`;
   - every kernel and const;
   - `Hparams` and `LayerKind` as the programs' input;
   - `KvLayout` and the placement internals;
   - the `roles.rs` rules;
   - the scattered file-vs-const checks and `pins()`. They stay as a second line: a file that passes the coverage check cannot trip them.

**The header test.** It is host-only; `Split::open` reads headers only. Under the wrap design, `Hparams::from_spec == Hparams::read` alone is close to a tautology, because the spec was built from that `Hparams`. What makes it a proof:
- **(a) Literal pins** for every §3 field, over four files: V4.1, Qwen3, V4-Flash and DSpark, each with a `PIN(date):` line.
- **(b) Consumer equality:**
  - `CedLayer::of(&spec.layers[l]) == CedLayer::of(&hp.layers[l])` for all 40 layers;
  - `KvLayout` from the spec equals `KvLayout::from_file`;
  - the projection `Hparams::from_spec == Hparams::read`, compared exactly (f32 by bits).
  (b) passes on the real files even though the refusal set moved by one case (index keys or a list on a window-only layer, §2a).
- **(c) The check's specimens:**
  - V4.1 and Qwen3 produce no items.
  - V4-Flash produces exactly `deepseek4_pins::UNIMPLEMENTED` (:184-204).
  - FAIL-first: with one instance removed from the available table, the check must list that instance's layers.
- **(d) Synthetic headers** (arch/mod.rs:172-239) must each give a named refusal:
  - a compressor at ratio 0;
  - index keys on a window-only layer (the new stricter case);
  - a `glm5next` header with no reader.

**Home.** The existing meta gates: `gate-ds41-meta`, `gate-qwen3moe-meta` and `gate-deepseek4-meta` (justfile:565-566, 570-571, 1162-1163). They take seconds and need no GPU.

**Proof class: move.**
- `just ptx-scan` is identical for every bin; no device code moves.
- The meta and placement gates are green.
- The changed host paths run their own gates (ds41-prefill, -step, -long, -faults, e2e), and the e2e set is identical.

### 7. Open questions

**For the lead:**
- **L1: crate placement.**
  - Recommendation: `crates/models` depends only on `gguf`, so plain cargo builds it.
  - The readers stay in `crates/model` for wave 3, because they wrap `Hparams`.
  - rebuild.md:62 puts the layer programs in `models`, but they need `ops`. Put them in their own crate or in `runtime` so that `models` stays host-only.
- **L2: wrap or rewrite.**
  - Recommendation: wrap in wave 3 and rewrite in wave 4.
  - The ≤ 300-line target fits new readers; qwen3moe is already 280 lines.
  - The deepseek reader's 1,259 lines are mostly refusals. Do not cut refusals to meet a line count.
- **L3: the V4-only fields.** Keep them (recommended), or add a reader-side `unsupported` list instead (§1).
- **L4: V4.1 above 16,384 positions.**
  - Recommendation: refuse `ctx_max > 16,384` by name, as llama.cpp does, until the candidate pool is built. Building it is size M.
  - This is a behaviour change, so it belongs in its own commit, not in the move.
  - The reader records `candidates: Some{20, 2048, 8}` from a cited constant (HF config.json:125-127).
- **L5: key strictness.**
  - Keep today's two policies: V4.1 takes no defaults; Qwen3 takes llama.cpp's defaults (qwen3moe/hparams.rs:107, 118-127).
  - New readers take the PR or mainline required set, cite each default's line, and print every default used on the load line.
  - `hash_layer_count` stays required. llama.cpp treats it as optional, but our file has it.
- **L6: unread keys.** Examples: Qwen3's `feed_forward_length` 6144, and `general.sampling.*` in both new files.
  - Recommendation: the meta gate lists them next to the roles' `Unread`. They are not refusals.
- **L7: tools and reasoning as spec fields.**
  - The reader sets them per family, and serve refuses a request with tools when `tools` is None.
  - Today only `bloomery-serve-ds41` binds a real engine; `bloomery-serve` is a mock (bloomery-serve.rs:1-9). So nothing served today changes.
- **L8: a Qwen3.6 arm in wave 3.**
  - A header-only `qwen35moe` arm (about 150 lines [estimate]) gives the check a live specimen, and the file is already on the box.
  - Recommendation: yes. The glm reader waits for wave 5.

**For 03 (kernelshape):**
- **K1: which keys are compile-time.**
  - Compile-time: route (E, K), GQA (HEAD, PACK), latent (LATENT, ROPE), delta rule (D, KDA), index_score (HEADS, D, Pool), hc STREAMS, draft MAX_WIDTH.
  - Runtime arguments: hidden (hc pieces, engram ROW), window, ratio, top-k, eps, scales, vocab.
  - NORM_K becomes a compile-time maximum with a runtime k.
- **K2: a host-readable instance table.** Emit it from the same macro that instantiates the kernels, in a crate plain cargo builds. The header test can then compute "needs minus instances" without the codegen backend.
- **K3: IMROPE for text.** With equal positions on every axis, does `Imrope{[11,11,10,0]}` over 64 of 256 dims equal partial NeoX? mistral.rs treats them as equal. Recommendation: check it once against a host reference before one kernel serves both.
- **K4: hc_pre formats.** `hc_pre<STREAMS, Fmt>` with Fmt ∈ {Q3_K, Q8_0, F32}. GLM needs Q8_0.
- **K5: the GLM collapse.** `ds41_hc_mean` computes `((s0+s1)+s2)+s3` then × 0.25. Does that match ik's "sum of streams × 1/hc" (build_glm5next.cpp:520-545) bit for bit?
- **K6: one `pool` op?** GLM's k-pool and V4's indexer compressor are both a softmax(gate + ape) pool, and they share tensor names (`indexer_compressor_{gate,ape}`). Their key paths differ: V4 uses `_kv` plus an RMS `_norm`; GLM uses `indexer.attn_k` plus a LayerNorm with bias. One op with two key paths may cover both. This is unverified beyond the tensor lists.

### 8. Improvement spots (report only)

| path:line | Spot | Size |
|---|---|---|
| generate_ds41 / gpu-deepseek41 (no cap found) | V4.1 above 16,384 positions departs from the reference without saying so (the candidate pool) | cap XS, build M |
| crates/serve/src/api.rs:1550-1555, reasoning.rs:32 | DSML and DeepSeek reasoning are applied to every template. The first non-DeepSeek binding (and the mock with `--model <qwen.gguf>` today) returns tool calls as plain content | S |
| crates/engram/src/hash.rs:41 | `KEY_PREFIX = "deepseek41."` hard-codes the architecture; use `split.arch_key`. plan-triage.md:290 lists the `RowTypes::engram()` twin | XS |
| crates/gpu-deepseek41/src/router.rs:280, 389 | the kernels return without writing when `n_expert ≠ N_EXPERT`. The host checks (:717, :780) make it unreachable, but it is a silent path by AGENTS' rule; raise the fault word instead | XS |
| gpu-deepseek41: chain/attn.rs:801-813, chain/ffn.rs:841-852, chain/glue.rs:611, body.rs:2346, indexer.rs:1006, attn.rs:951-955 and 1024-1030, hc.rs:1526-1531, draft/load.rs:274-280, draft/block.rs:188 | ten file-vs-const checks, each stopping at its first mismatch. `pins()` (qwen3moe/body.rs:120-155) collects all of its mismatches. The coverage check replaces both | S (waves 3-4) |
| crates/model/src/arch/qwen3moe/hparams.rs:102-140 | `feed_forward_length` 6144 is neither read nor reported | XS |
| crates/model/src/arch/deepseek41/roles.rs:39-41; docs/rebuild.md:156, :159 | the nextn rules match nothing in our file; the plan says the three layers are `Unused` today; "Q5_K routed" belongs to Qwen3.6 | XS doc |
| crates/model/src/arch/deepseek41/hparams.rs:426 | `hash_layer_count` is required; llama.cpp makes it optional because V4.1 files may omit it | XS |
| ik src/llama-hparams.cpp:2423 | the comment says the clamp is before the SiLU; the kernel clamps after (unary.cu:80-82). Upstream comment fix; check for duplicates first | XS upstream |
| tools (plan-triage.md:290) | this round again needed a header dumper that prints tensor types by layer range. Promote it to `gguf-inventory --keys --layers` | S |
| tools/box.sh:63 | READONLY still rsyncs (already in the boxlease triage) | XS |

### 9. What I could not read

- The GLM and Qwen3.6 chat templates beyond their first 120 characters. Their tool-call and thinking syntax is unverified.
- Whether Qwen3.6-35B-A3B's GDN output gate is SiLU:
  - exllamav3's `GatedDeltaNet` call (qwen3_5.py:330-353) passes no gate type;
  - modelvocab A gives `output_gate_type "swish"` only for 3.6-27B.
- exllamav3's `act_limit` rule.
- ggml's IMROPE code (K3).
- Whether ik supports qwen35moe.
- PR #27752 and llama.cpp's GLM MTP PR #27917.
- ik's GLM graph beyond the cited lines, and the PR's graph beyond the grep.
- The V4-Flash header this round. Its facts come from `deepseek4_pins.rs`.
- **Side effect.** `tools/box.sh:63` rsyncs even under `BLOOMERY_BOX_READONLY=1`. My READONLY calls synced the lead's main tree (clean, `a23b0ba`) to the default remote directory.
- **Temporary files.** The header dumps this memo quotes are kept outside the repository; `gguf-inventory --keys --layers` (§8) would make them reproducible from the tree.

### 10. Model

Opus 5.5 (`claude-opus-5-5`), at the round definition's effort.
