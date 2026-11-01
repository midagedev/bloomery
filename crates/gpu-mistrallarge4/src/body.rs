//! The Mistral Large 4 whole-card body: one [`Body`] behind [`ChainBody`],
//! every tensor resident on one card, the step one token's. The launch order
//! is the reference graph's (`src/models/mistral-large4.cpp`, PR #31376):
//!
//! 1. the embedding row of the token (`embed_rows_q6k`), the running row the
//!    whole chain reads;
//! 2. `attn_norm` and its q8_1 form in one launch (`norm_quant`);
//! 3. `attn_q` (Q6_K) of the normed row — a head's `[nope | rope]` values —
//!    and the rope of every 64-value column of it (`rope`; a rotated nope
//!    column is never read);
//! 4. on a layer that owns its cache, `attn_kv_a_mqa` (Q8_0) of the normed
//!    row, then the fused key path (`kv_norm_rope_append`): the latent's RMS
//!    norm, the rope of the row's tail, the `[k_rope | kv_compressed]` f16
//!    row appended at the position — a NoPE owner appends its bare latent
//!    (`latent_append`), and a reader layer appends nothing;
//! 5. on a full-indexer layer, the lightning indexer: the LayerNorm of the
//!    normed row (`ln_rows`) and `indexer.attn_q_b` (Q6_K) of it, the file's
//!    `[rope | nope]` head rows moved to the kernel's tail order by the
//!    permutation of the projection's output (`gather_pairs`; the weight
//!    itself stays as the file holds it), the index queries' SSMax
//!    (`ssmax_rows`), `indexer.attn_k` (Q8_0) of the normed row and the same
//!    permutation of it, the key's LayerNorm, tail rope, Hadamard rotation
//!    and f16 store at the position (`ds41_index_key_input` — the k-norm's
//!    gain and bias moved at load the same way), `indexer.proj` (F32) of the
//!    normed row, the scores and the list (`ds41_indexer_score`,
//!    `ds41_indexer_topk`); a NoPE layer's index head has no rope rows, so
//!    neither permutation runs there and its k-norm pair stays as the file
//!    holds it; a shared-indexer layer launches none of these and reads the
//!    layer it names;
//! 6. the absorbed query: `attn_k_b` (Q8_0) dots each head's nope slice of
//!    `attn_q`'s row into the latent width (`q8_0_gemv_heads`), a rope
//!    layer's flash row `[q_rope | q_nope2]` assembled by `gather_pairs`
//!    from the roped columns, then the main SSMax over the whole row
//!    (`ssmax_rows` — the reference scales the joined row, rope part
//!    included); a NoPE layer's absorbed row is the query whole;
//! 7. the attention: a rope layer gathers the list's rows of the cache into
//!    a contiguous scratch (`kv_gather_sel`) and runs the dense latent flash
//!    over them (`flash_latent` — one segment — else `flash_latent_mma` and
//!    `flash_merge`), the scale `1/√key_length_mla`; a NoPE layer runs the
//!    512-wide walk over its cache through the list (`ds41_attn_seg_sel`,
//!    `ds41_attn_merge`) with no window and a −∞ sink a head;
//! 8. `attn_v_b` (Q8_0) up-projects each head's latent output
//!    (`q8_0_gemv_heads`), `attn_output` (Q6_K) projects the heads' values,
//!    and the residual add makes `ffn_inp`;
//! 9. `ffn_norm`, then the dense lead's gate·up·down (Q8_0, `swiglu`) or the
//!    routed block: the router's logits (`f32_gemv` of `ffn_gate_inp`'s rows
//!    — a file of fewer experts than the routing body's 128 lanes carries
//!    the pads it never picks, `f32::MIN`, written once at load) and the
//!    softmax top-k routing (`softmax_router_route`: the kept weights
//!    renormalized, the chosen probabilities divided by their sum), the
//!    i-quant gate·up and down of the picked experts
//!    (`*_gate_up_sel`, `*_gemv_sel` at the file's own type), their weighted
//!    sum, and the shared expert's gate·up·down beside it, then the two
//!    residual adds;
//! 10. the head: the final RMS norm, the Q6_K `output` projection, the
//!     tanh softcap of the file's bound and the argmax
//!     (`Head::enqueue_with_tail`).
//!
//! Every count the chain reads — the position, the live rows, the list's
//! width, the walk's visible pair, the rope row — lives in one of five small
//! device buffers the host refreshes before a replay (the indexer's own
//! `n_vis` word written by the embedding launch itself), so one capture
//! serves every position. The router's pad lanes keep their load-time value:
//! the launches never write them.
//!
//! The taps (`set_taps`) hold a copy of every compared value a layer leaves,
//! read back by a gate; they refuse a captured graph, which would not hold
//! the copies.

use bloomery_gpu::Pitched;
use bloomery_gpu::arch::hybrid::{Acts, DecodeInput, GqaCx};
use bloomery_gpu::flash::{FlashKernels, FlashMergeArgs, FlashSegArgs, segments_for};
use bloomery_gpu::flash_gqa::FlashGqaKernels;
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::head::{Head, HeadNorm};
use bloomery_gpu::iq_sel::IqGateUp;
use bloomery_gpu::kquant::{Act as GateAct, SelDown};
use bloomery_gpu::latent::{KvGatherSelArgs, LatentAppendArgs, LatentKernels, Rows as LatentRows};
use bloomery_gpu::model::{ChainBody, GpuModel, Q8_0GemvHeadsArgs, StepKernels};
use bloomery_gpu::quant_sel::QuantSelKernels;
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::route_softmax::{RouterBufs, RouterDims, RouterKernels};
use bloomery_gpu::site::KGemvKernels;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, FaultSink, FaultSite, Gpu, GpuError, Q8Act, Window, dtod_2d};
use bloomery_gpu_deepseek41::attn::{AttnArgs, AttnKernels, SelectedRows, segments};
use bloomery_gpu_deepseek41::index_key::{IndexKeyInputArgs, IndexKeyKernels};
use bloomery_gpu_deepseek41::indexer::{IndexerArgs, IndexerKernels, IndexerScratch};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::mistrallarge4::hparams::Hparams;
use model::arch::names::token_embd;
use models::shape::{MoeShape, rules};
use std::ops::Range;

use crate::tensors::{self, Ffn, Kind, Layer};

const WHAT: &str = "mistrallarge4::Body::load";
const STEP: &str = "mistrallarge4::Body::enqueue_chain";

/// The experts the routing body softmaxes over: the qwen3moe instance's own
/// width, the real file's expert count. A file of fewer experts (the
/// fixture's 16) routes through it on pad lanes the picks never reach.
const ROUTE_WIDTH: usize = 128;

/// The f32 that fills the routing body's pad lanes: finite, and its
/// probability `exp(f32::MIN − max)` is exactly zero, so a pad lane never
/// enters the top-k — every real lane's probability is at least zero at a
/// smaller id, and the tie rule keeps the smaller id.
const ROUTE_PAD: f32 = f32::MIN;

/// The step's word buffers, refreshed before a replay: the position (read at
/// word 0 by every launch that takes a position), the indexer's counts
/// (`n_vis` written by the embedding launch, `top_k` once at load, the token
/// beside them), the walk's visible pair and the selected attention's count.
mod words {
    /// The position, word 0 of its own buffer.
    pub(crate) const POS: usize = 0;
    /// `ints[0]`: the live rows, the embedding launch's own `n_keys` write.
    pub(crate) const N_VIS: usize = 0;
    /// `ints[1]`: the list's width, written once at load.
    pub(crate) const TOP_K: usize = 1;
    /// The selected attention's count, word 0 of its own buffer.
    pub(crate) const N_KEYS: usize = 0;
}

/// The kernels the step launches beyond the `Gpu`'s own.
struct Kernels {
    fused: FusedKernels,
    latent: LatentKernels,
    flash: FlashKernels,
    attn: AttnKernels,
    index_key: IndexKeyKernels,
    /// The lightning indexer at the file's two rope widths: a roped layer's
    /// `rope.dimension_count`, and 0 on a NoPE layer's index head.
    indexer_rope: IndexerKernels,
    indexer_nope: IndexerKernels,
    router: RouterKernels,
    iq: QuantSelKernels,
    step: StepKernels,
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash_gqa: FlashGqaKernels,
}

impl Kernels {
    fn load(gpu: &Gpu, sh: &Shape) -> Result<Kernels, GpuError> {
        let (heads, dim, rope) = (sh.idx_heads, sh.idx_d, sh.rope);
        let ctx = gpu.context();
        Ok(Kernels {
            fused: FusedKernels::load(ctx)?,
            latent: LatentKernels::load(ctx)?,
            flash: FlashKernels::load(ctx)?,
            attn: AttnKernels::load(ctx)?,
            index_key: IndexKeyKernels::load(ctx)?,
            indexer_rope: IndexerKernels::with_shape(ctx, heads, dim, rope)?,
            indexer_nope: IndexerKernels::with_shape(ctx, heads, dim, 0)?,
            router: RouterKernels::load(ctx)?,
            iq: QuantSelKernels::load(ctx, gpu.fault_word())?,
            step: StepKernels::load(ctx)?,
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash_gqa: FlashGqaKernels::load(ctx)?,
        })
    }
}

/// The widths the step's buffers are cut for, from the file's own keys.
#[derive(Clone, Copy)]
struct Shape {
    hidden: usize,
    heads: usize,
    /// `attention.key_length_mla`: a query head's whole width.
    head_k: usize,
    head_v: usize,
    /// `attention.kv_lora_rank`: the cached latent's width.
    latent: usize,
    /// `rope.dimension_count`.
    rope: usize,
    /// `attention.indexer.head_count` and `key_length`: the lightning
    /// indexer's own heads of values.
    idx_heads: usize,
    idx_d: usize,
    /// `attention.indexer.top_k`: the list's width.
    top_k: usize,
    experts: usize,
    used: usize,
    ff: usize,
    /// The dense lead's width, the widest gate·up the step runs.
    dense_ff: usize,
    n_shared: usize,
    eps: f32,
    /// The indexer's LayerNorms' own epsilon.
    idx_eps: f32,
    ctx: usize,
    n_layer: usize,
    softcap: f32,
}

impl Shape {
    /// The rope part of a cache row: a roped layer's `rope`, none on NoPE.
    fn pe(&self, rope_layer: bool) -> usize {
        if rope_layer { self.rope } else { 0 }
    }
}

/// A full indexer layer's stores.
struct IdxStore {
    keys: DeviceTensor<u16>,
    list: DeviceBuffer<u32>,
    scratch: IndexerScratch,
}

/// The step's own buffers, allocated once at load.
struct Scratch {
    xn: DeviceBuffer<f32>,
    x_ln: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    /// A rope layer's flash rows, a NoPE layer's absorbed queries.
    q_cur: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    qi: DeviceBuffer<f32>,
    /// The index queries in the kernel's head order, the gather's own.
    qi_moved: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    /// The key projection, the file's head order and the kernel's.
    kidx: DeviceBuffer<f32>,
    kidx_moved: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    /// The dense lead's and the shared expert's gate, up and rule output,
    /// the widest of the two.
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    rows: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    /// The rope layer's gathered cache rows, `top_k` of the cache's width.
    gathered: DeviceTensor<u16>,
    pos: DeviceBuffer<u32>,
    /// The token's own word, the embedding lookup's id.
    token: DeviceBuffer<u32>,
    /// The embedding's redundant position write (`pos0 + first`, the same
    /// value the refresh wrote), kept out of `pos` so the launch reads one
    /// and writes the other.
    pos_out: DeviceBuffer<u32>,
    ints: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// The gathered flash's own row count, `min(pos, top_k − 1)`: the flash
    /// counts the current row on top of the word it reads, and the gathered
    /// rows already hold it (the selection is of `pos + 1` keys).
    flash_n: DeviceBuffer<u32>,
    cs: DeviceBuffer<f32>,
    /// The rope row's f32 bits as u32 words, the score pass's own table.
    table_words: DeviceBuffer<u32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    host_pos: Vec<u32>,
    host_token: Vec<u32>,
    host_ints: Vec<u32>,
    host_vis: Vec<u32>,
    host_n_keys: Vec<u32>,
    host_flash_n: Vec<u32>,
    host_cs: Vec<f32>,
    host_table_words: Vec<u32>,
}

/// The load-time index tables of the two permutations the step gathers: the
/// index head's `[rope | nope]` file order as the kernels' `[nope | rope]`
/// tail order, one table the queries' projection width and one the key's,
/// and the flash row's rope spans out of the roped columns.
struct Perm {
    /// `(src, dst)` over the queries' projection, a head's block at a time.
    q: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    /// The key projection's own, one head.
    k: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    /// The flash rows' rope spans: each head's roped tail column into the
    /// first `rope` values of its row.
    f_rope: (DeviceBuffer<u32>, DeviceBuffer<u32>),
}

/// The body: the shape, each layer's tensors and stores, the kernels, the
/// activation forms and the step's buffers.
/// One layer's compared values, read back after a step (a gate's
/// instrument; the oracle's own `cb()` names).
pub struct LayerNodes {
    pub attn_norm: Vec<f32>,
    /// The absorbed query after its SSMax scale: a rope layer's flash row,
    /// a NoPE layer's latent-width query.
    pub q_cur: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub l_out: Vec<f32>,
    /// The latent projection and its normed, permuted form (`kv_cmpr`'s two
    /// occurrences; a NoPE owner's rope part is its zero tail).
    pub kv_a: Vec<f32>,
    pub kv_s: Vec<f32>,
    /// The appended key row, `[k_rope | kv_compressed]` order.
    pub kvr: Vec<f32>,
    /// The index queries after their SSMax scale, the key projection in the
    /// kernel's head order and the head weights.
    pub qi: Vec<f32>,
    pub kidx: Vec<f32>,
    pub wi: Vec<f32>,
    pub q: Vec<f32>,
    pub q_rope: Vec<f32>,
    /// The layer's list, its source's on a shared layer.
    pub list: Vec<u32>,
    /// The layer's own latent cache, f16 as written (an owner's).
    pub cache16: Vec<u16>,
    /// The attention block's output row before the value up-projection.
    pub att: Vec<f32>,
    /// The heads' values up-projected (the reference's `fattn_mla`).
    pub av: Vec<f32>,
    pub moe: Vec<f32>,
    pub sh: Vec<f32>,
    pub ffn_out: Vec<f32>,
    /// The routing of a routed layer: the real lanes' logits, the kept ids,
    /// their probabilities and their renormalized weights.
    pub route_logits: Vec<f32>,
    pub route_ids: Vec<u32>,
    pub route_probs: Vec<f32>,
    pub route_weights: Vec<f32>,
}

pub struct Body {
    sh: Shape,
    layers: Vec<Layer>,
    stores: Vec<LayerStore>,
    k: Kernels,
    acts: Acts,
    perm: Perm,
    /// The routed layers' routing results, in layer order.
    routes: Vec<Option<RouterBufs>>,
    /// The experts' own activation forms: the normed row a token (the
    /// gate·up's input) and the gate·up output a kept slot.
    act_hidden: Q8Act,
    act_ff: Q8Act,
    s: Scratch,
    table: RopeTable,
    taps: Option<Taps>,
}

/// A layer's stores: its latent cache (an owner's), the index-key cache, the
/// list and the scratch (a full indexer's own).
struct LayerStore {
    /// `[ctx × (latent + pe)]` f16, `None` on a reader layer.
    latent: Option<DeviceTensor<u16>>,
    idx: Option<IdxStore>,
}

/// The per-layer staging a gate's taps read, one a layer.
struct Taps {
    layers: Vec<LayerTap>,
    /// The rows that replace the running row ahead of their layer, at most
    /// one a layer.
    inject: Vec<(usize, DeviceBuffer<f32>)>,
    span: Option<Range<usize>>,
}

/// One layer's compared values, copied as the layer leaves them.
struct LayerTap {
    attn_norm: DeviceBuffer<f32>,
    q_cur: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    l_out: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    qi: DeviceBuffer<f32>,
    kidx: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    /// The query projection and its roped columns (the flash row's rope span
    /// out of them).
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    /// The layer's list, its source's on a shared layer.
    list: DeviceBuffer<u32>,
    /// The whole latent cache of the layer's own store, f16 as written (an
    /// owner's; a reader's stays zero): the NoPE owner's appended row has no
    /// f32 form outside the append.
    cache16: DeviceBuffer<u16>,
    /// The attention block's output row before the value up-projection
    /// (`heads × latent`).
    att: DeviceBuffer<f32>,
    /// The heads' values up-projected (`heads × head_v`, the reference's
    /// `fattn_mla`).
    av: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    /// The layer ropes: its index taps copy the moved rows, a NoPE layer's
    /// the projections as written.
    rope: bool,
}

impl Body {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format, the latent and index-key
    /// caches of `ctx` rows, the lists, the routers' results, the rope table
    /// and the step's buffers. Refused by name: a context past the trained
    /// one, a head or key width the kernels do not take, an expert
    /// count past the routing body's lanes, and every tensor-shape refusal
    /// the sites and stacks make.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layer = Hparams::read(&file)
            .map_err(|e| GpuError::plan(WHAT, e))?
            .n_layer;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layer, true, |gpu, w| {
            Body::load(gpu, &file, w, ctx)
        })
    }
}

impl Body {
    /// The body of a whole-card load.
    fn load(gpu: &Gpu, file: &Split, w: &Weights, ctx: usize) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        // The context bound is the reader's own rule: the trained context
        // (the selected attention serves past the indexer's top-k).
        let read =
            model::arch::mistrallarge4::spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        model::arch::mistrallarge4::spec::serve_ctx(ctx as u64, &read.spec)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let sh = Shape {
            hidden: hp.n_embd,
            heads: hp.n_head,
            head_k: hp.head_k,
            head_v: hp.head_v,
            latent: hp.kv_lora,
            rope: hp.rope_dims,
            idx_heads: hp.indexer_heads,
            idx_d: hp.indexer_d,
            top_k: hp.indexer_top_k,
            experts: hp.n_expert,
            used: hp.n_used,
            ff: hp.expert_ff,
            dense_ff: hp.dense_ff,
            n_shared: hp.n_shared,
            eps: hp.rms_eps,
            idx_eps: hp.idx_norm_eps,
            ctx,
            n_layer: hp.n_layer,
            softcap: hp.softcap,
        };
        // The lightning indexer's own head counts are the kernel's.
        if sh.idx_heads != bloomery_gpu_deepseek41::indexer::HEADS
            || sh.idx_d != bloomery_gpu_deepseek41::indexer::HEAD_DIM
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the indexer has {} heads of {}, the kernels run {} of {}",
                    sh.idx_heads,
                    sh.idx_d,
                    bloomery_gpu_deepseek41::indexer::HEADS,
                    bloomery_gpu_deepseek41::indexer::HEAD_DIM
                ),
            ));
        }
        if hp.n_expert > ROUTE_WIDTH {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the file routes {} experts, the routing body serves at most {ROUTE_WIDTH}",
                    hp.n_expert
                ),
            ));
        }
        if sh.top_k == 0 || sh.used == 0 || sh.used > sh.top_k {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the list's width {} and the router's {} picks: at least one pick, at most \
                     the list's width",
                    sh.top_k, sh.used
                ),
            ));
        }
        let stream = gpu.stream();
        let layers = (0..hp.n_layer)
            .map(|l| tensors::layer((file, w), &hp, stream, l))
            .collect::<Result<Vec<_>, _>>()?;
        let stores = layers
            .iter()
            .enumerate()
            .map(|(l, a)| {
                let width = sh.latent + sh.pe(a.kind.rope);
                Ok(LayerStore {
                    latent: if a.kind.cache == l {
                        Some(DeviceTensor::zeroed(stream, ctx, width)?)
                    } else {
                        None
                    },
                    idx: if a.kind.full {
                        Some(IdxStore {
                            keys: DeviceTensor::zeroed(stream, ctx, sh.idx_d)?,
                            list: DeviceBuffer::zeroed(stream, sh.top_k)?,
                            scratch: IndexerScratch::new(stream, 1, ctx)?,
                        })
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let k = Kernels::load(gpu, &sh)?;
        let acts = Acts::new(stream, layers.iter().flat_map(tensors::sites))?;
        let act_hidden = Q8Act::with_k(stream, 1, sh.hidden)?;
        let act_ff = Q8Act::with_slots(stream, sh.used, sh.ff)?;
        // The routers' buffers, one a routed layer, its pad lanes written
        // once: the launches never touch them.
        let dims = RouterDims::of(MoeShape {
            rule: rules::SOFTMAX_NORM,
            experts: ROUTE_WIDTH as u32,
            top_k: sh.used as u32,
        })
        .map_err(|e| GpuError::plan(WHAT, e))?;
        let mut routes = Vec::with_capacity(layers.len());
        let pad: Vec<f32> = std::iter::repeat_n(0.0, sh.experts)
            .chain(std::iter::repeat_n(ROUTE_PAD, ROUTE_WIDTH - sh.experts))
            .collect();
        for a in &layers {
            if matches!(a.ffn, Ffn::Moe { .. }) {
                let mut route = RouterBufs::with_tokens(stream, dims, 1)?;
                route.logits.copy_from_host(stream, &pad)?;
                routes.push(Some(route));
            } else {
                routes.push(None);
            }
        }
        let mut s = Scratch::new(stream, &sh)?;
        let perm = Perm::new(stream, &sh)?;
        // The counts that never move: the list's width, beside the live rows
        // the embedding launch writes and the token.
        s.ints.copy_from_host(stream, &[0, sh.top_k as u32])?;
        // The rope table: YaRN at the LeRoPE factors, the attention factor
        // the form computes cancelling ggml's own magnitude scale (the
        // reader's `Yarn.magnitude` states the net, 1). The factors are the
        // file's own global tensor.
        let spec = RopeSpec::yarn(
            hp.rope_base,
            hp.yarn.factor,
            hp.yarn.orig_ctx as i32,
            hp.yarn.beta_fast,
            hp.yarn.beta_slow,
            sh.rope,
        )
        .with_factors(&tensors::file_f32(file, "rope_freqs.weight", sh.rope / 2)?)
        .map_err(|e| GpuError::plan(WHAT, e))?;
        let table = RopeTable::new(&spec).map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(Body {
            sh,
            layers,
            stores,
            k,
            acts,
            perm,
            routes,
            act_hidden,
            act_ff,
            s,
            table,
            taps: None,
        })
    }

    /// The whole chain (the module doc's ten stages): the embedding, every
    /// layer of the span the taps hold (all of them when none does), the
    /// head when the span ends at the model's end.
    fn enqueue(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let Body {
            sh,
            layers,
            stores,
            k,
            acts,
            perm,
            routes,
            act_hidden,
            act_ff,
            s,
            table: _,
            taps,
        } = self;
        let sh = *sh;
        let cx = GqaCx {
            what: STEP,
            gpu,
            w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash_gqa,
            mma: false,
            out_gate: None,
            hidden: sh.hidden,
            eps: sh.eps,
            n_kv: 1,
            ctx: sh.ctx,
        };
        let stream = gpu.stream();
        let elem = gpu.elem();
        // 1. the embedding row, the position and the live count the
        //    launches below read (the position written back unchanged, the
        //    count the indexer's own `n_vis` word).
        {
            let DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w: table,
                ..
            } = w.resident(STEP, &token_embd())?
            else {
                return Err(GpuError::tensor(
                    STEP,
                    token_embd(),
                    "a Q6_K embedding table",
                ));
            };
            let ids = Window::<u32>::of(&s.token, 0, 1)?;
            elem.enqueue_embed_rows_kquant(
                stream,
                GgmlType::Q6_K,
                bloomery_gpu::elem::EmbedRowsArgs {
                    w: table,
                    ids: &ids,
                    pos0: &s.pos,
                    first: 0,
                    y: head.input_mut(),
                    pos: &mut s.pos_out,
                    n_keys: &mut s.ints,
                },
            )?;
        }
        let span = taps
            .as_ref()
            .and_then(|t| t.span.clone())
            .unwrap_or(0..sh.n_layer);
        for l in span.clone() {
            if let Some((_, row)) = taps
                .as_ref()
                .and_then(|t| t.inject.iter().find(|(at, _)| *at == l))
            {
                copy_row(stream, row, head.input_mut(), sh.hidden)?;
            }
            attention(
                gpu,
                w,
                k,
                sh,
                &cx,
                &layers[l],
                l,
                stores,
                acts,
                perm,
                s,
                head.input(),
                gpu.layer_sink(l)?,
            )?;
            // The attention's norm row: the ffn's own `rms_norm` reuses the
            // same scratch, so its tap copies before the ffn runs.
            if let Some(t) = taps.as_mut() {
                t.layers[l].snap_attention(stream, s, sh)?;
            }
            ffn(
                gpu,
                k,
                sh,
                &cx,
                (&layers[l], l),
                routes,
                acts,
                (act_hidden, act_ff),
                s,
                head.input_mut(),
            )?;
            if let Some(t) = taps.as_mut() {
                let list = stores[layers[l].kind.indexer]
                    .idx
                    .as_ref()
                    .map(|st| &st.list);
                let cache = (layers[l].kind.cache == l)
                    .then(|| stores[l].latent.as_ref())
                    .flatten();
                t.layers[l].snap(stream, s, head.input(), list, cache, sh)?;
            }
        }
        if span.end == sh.n_layer {
            let softcap = sh.softcap;
            let n_vocab = head.n_vocab();
            head.enqueue_with_tail(gpu, w, |act, out_w, fault, logits, token_out| {
                gpu.enqueue_gemv_q6k(out_w, act, logits)?;
                // 0 is no cap: the reference graph caps only on a nonzero bound.
                if softcap > 0.0 {
                    elem.enqueue_softcap_rows(gpu.stream(), logits, n_vocab, softcap)?;
                }
                elem.enqueue_argmax_fault(gpu.stream(), logits, n_vocab, fault, token_out)
            })?;
        }
        Ok(())
    }
}

impl Scratch {
    /// The step's buffers, `hidden`-wide rows and the attention partials at
    /// the wider of the flash's segments over the scratch and the walk's
    /// over the context.
    fn new(stream: &CudaStream, sh: &Shape) -> Result<Scratch, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let width = sh.latent + sh.rope;
        let (idx_q, idx_d) = (sh.idx_heads * sh.idx_d, sh.idx_d);
        let wide_ff = sh.dense_ff.max(sh.n_shared * sh.ff);
        let q_rows = sh.heads;
        let segs = segments_for(sh.top_k).max(segments(0, sh.ctx));
        let sinks: Vec<f32> = std::iter::repeat_n(f32::NEG_INFINITY, sh.heads).collect();
        Ok(Scratch {
            xn: z(sh.hidden)?,
            x_ln: z(sh.hidden)?,
            q: z(sh.heads * sh.head_k)?,
            q_rope: z(sh.heads * sh.head_k)?,
            q_cur: z(sh.heads * width)?,
            kv_a: z(width)?,
            kv_s: z(width)?,
            kvr: z(width)?,
            qi: z(idx_q)?,
            qi_moved: z(idx_q)?,
            wi: z(sh.idx_heads)?,
            kidx: z(idx_d)?,
            kidx_moved: z(idx_d)?,
            att: z(sh.heads * sh.latent)?,
            av: z(sh.heads * sh.head_v)?,
            r: z(sh.hidden)?,
            ffn_inp: z(sh.hidden)?,
            g: z(wide_ff)?,
            u: z(wide_ff)?,
            act: z(wide_ff)?,
            up: z(sh.used * sh.ff)?,
            rows: z(sh.used * sh.hidden)?,
            moe: z(sh.hidden)?,
            sh: z(sh.hidden)?,
            ffn_out: z(sh.hidden)?,
            gathered: DeviceTensor::zeroed(stream, sh.top_k, width)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            token: DeviceBuffer::zeroed(stream, 1)?,
            pos_out: DeviceBuffer::zeroed(stream, 1)?,
            ints: DeviceBuffer::zeroed(stream, 2)?,
            vis: DeviceBuffer::zeroed(stream, 2)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
            flash_n: DeviceBuffer::zeroed(stream, 1)?,
            cs: DeviceBuffer::zeroed(stream, sh.rope)?,
            table_words: DeviceBuffer::zeroed(stream, sh.rope)?,
            part_v: z(q_rows * segs * sh.latent)?,
            part_ms: z(q_rows * segs * 2)?,
            sinks: DeviceBuffer::from_host(stream, &sinks)?,
            host_pos: vec![0; 1],
            host_token: vec![0; 1],
            host_ints: vec![0, 0],
            host_vis: vec![0; 2],
            host_n_keys: vec![0; 1],
            host_flash_n: vec![0; 1],
            host_cs: vec![0.0; sh.rope],
            host_table_words: vec![0; sh.rope],
        })
    }
}

impl Perm {
    /// The three tables: the index head's move (a query projection wide, and
    /// the key's one head) and the flash row's rope spans.
    fn new(stream: &CudaStream, sh: &Shape) -> Result<Perm, GpuError> {
        let table = |pairs: Vec<(usize, usize)>| -> Result<
            (DeviceBuffer<u32>, DeviceBuffer<u32>),
            GpuError,
        > {
            let (src, dst): (Vec<u32>, Vec<u32>) =
                pairs.into_iter().map(|(s, d)| (s as u32, d as u32)).unzip();
            Ok((
                DeviceBuffer::from_host(stream, &src)?,
                DeviceBuffer::from_host(stream, &dst)?,
            ))
        };
        // Within each `idx_d`-value head, the file's `[rope | nope]` as the
        // kernel's tail order: value `j` of the moved head reads the file's
        // `j + rope` while it fits the nope part, else `j - (idx_d - rope)`.
        let head_move = |head: usize, rope: usize| {
            (0..head).map(move |j| {
                (
                    if j < head - rope {
                        j + rope
                    } else {
                        j - (head - rope)
                    },
                    j,
                )
            })
        };
        let q = table(
            (0..sh.idx_heads)
                .flat_map(|h| {
                    head_move(sh.idx_d, sh.rope)
                        .map(move |(s, j)| (h * sh.idx_d + s, h * sh.idx_d + j))
                })
                .collect(),
        )?;
        let k = table(head_move(sh.idx_d, sh.rope).collect())?;
        // Each head's roped tail column — the last `rope` values of its
        // `head_k`-wide slice — into the first `rope` values of its flash
        // row, the absorbed span behind them.
        let f_rope = table(
            (0..sh.heads)
                .flat_map(|h| {
                    (0..sh.rope).map(move |j| {
                        (
                            h * sh.head_k + (sh.head_k - sh.rope) + j,
                            h * (sh.latent + sh.rope) + j,
                        )
                    })
                })
                .collect(),
        )?;
        Ok(Perm { q, k, f_rope })
    }
}

/// `n` f32 of `src` copied into `dst`, one row.
fn copy_row(
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    n: usize,
) -> Result<(), GpuError> {
    let width = n * 4;
    dtod_2d(
        stream,
        (src, 0),
        (dst, 0),
        Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )
}

/// The attention's logit scale: `1/√key_length_mla`, the reference's own
/// (`kq_scale` of its graph — YaRN without magnitude scaling).
fn kq_scale(sh: Shape) -> f32 {
    1.0 / (sh.head_k as f32).sqrt()
}

/// The resident i-quant stack `name`'s word tensor.
fn stack<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(STEP, name)? {
        DevWeight::KQuant { w: t, .. } => Ok(t),
        _ => Err(GpuError::tensor(STEP, name, "an i-quant stack")),
    }
}

/// The attention half of layer `l` (the module doc's stages 2–8): `x` in,
/// the residual's delta into `s.r`.
#[allow(
    clippy::too_many_arguments,
    reason = "the half's launches: its card, weights and kernels, the shape, the site context, the layer and its number, the stores, the activation forms, the permutations, the buffers, the running row and the fault sink (rust-quality R8)"
)]
fn attention(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    sh: Shape,
    cx: &GqaCx<'_>,
    a: &Layer,
    l: usize,
    stores: &mut [LayerStore],
    acts: &mut Acts,
    perm: &Perm,
    s: &mut Scratch,
    x: &DeviceBuffer<f32>,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let (rope, pe) = (a.kind.rope, sh.pe(a.kind.rope));
    let width = sh.latent + pe;
    // 2. the norm and its q8_1 form, one launch.
    let act = acts.q8_1_at(STEP, sh.hidden, 1)?;
    k.fused.enqueue_norm_quant(
        stream,
        x,
        w.f32_buf(STEP, &a.attn_norm)?,
        sh.eps,
        act,
        &mut s.xn,
        fault,
    )?;
    // 3. the query projection and the rope of its columns.
    a.q.kgemv(cx, act, l, &mut s.q)?;
    if rope {
        elem.enqueue_rope(
            stream,
            &s.q,
            &s.cs,
            sh.rope,
            (sh.heads * sh.head_k / sh.rope) as u32,
            1,
            &mut s.q_rope,
        )?;
    }
    // 4. the key path: the latent projection, then the fused norm, rope and
    //    f16 append of the `[k_rope | kv_compressed]` row — the bare
    //    latent's own append on a NoPE owner, nothing on a reader.
    if a.kind.cache == l {
        let kv_a = a.kv_a.as_ref().ok_or_else(|| GpuError::State {
            what: STEP,
            missing: "the kv_a site",
        })?;
        kv_a.project(cx, acts, l, &s.xn, &mut s.kv_a)?;
        let norm = w.f32_buf(
            STEP,
            a.kv_a_norm.as_ref().expect("an owner carries the norm"),
        )?;
        let st = &mut stores[l];
        let cache = st.latent.as_mut().ok_or_else(|| GpuError::State {
            what: STEP,
            missing: "the layer's latent store",
        })?;
        if rope {
            k.fused.enqueue_kv_norm_rope_append(
                stream,
                &s.kv_a,
                norm,
                &s.cs,
                &s.pos,
                sh.eps,
                sh.latent,
                sh.rope,
                &mut s.kv_s,
                &mut s.kvr,
                cache,
                fault,
            )?;
        } else {
            k.latent.enqueue_latent_append(
                stream,
                LatentAppendArgs {
                    rows: LatentRows {
                        x: &s.kv_a,
                        stride: sh.latent,
                        m: 1,
                    },
                    off: 0,
                    gain: norm,
                    pos: &s.pos,
                    eps: sh.eps,
                    fault,
                    cache,
                },
            )?;
        }
    }
    // 5. the lightning indexer: a full layer's keys, scores and list, in the
    //    reference's order (the module doc's stage 5).
    if a.kind.full {
        let idx = a.idx.as_ref().expect("a full layer carries its indexer");
        let st = stores[l].idx.as_mut().expect("a full layer's store");
        elem.enqueue_ln_rows(
            stream,
            &s.xn,
            w.f32_buf(STEP, &idx.q_norm_w)?,
            w.f32_buf(STEP, &idx.q_norm_b)?,
            sh.idx_eps,
            sh.hidden,
            1,
            fault,
            &mut s.x_ln,
        )?;
        // A roped layer's index head moves its leading rope rows to the
        // kernel's tail (the k-norm pair moved the same way at load,
        // `tensors::moved_k_norm`); a NoPE head has no rope rows, so its
        // projections stay as written and the gathers do not run.
        idx.q_b.project(cx, acts, l, &s.x_ln, &mut s.qi)?;
        if rope {
            k.step.enqueue_gather(
                stream,
                &s.qi,
                &perm.q.0,
                &perm.q.1,
                sh.idx_heads * sh.idx_d,
                &mut s.qi_moved,
            )?;
        }
        elem.enqueue_ssmax_rows(
            stream,
            if rope { &mut s.qi_moved } else { &mut s.qi },
            &s.pos,
            w.f32_buf(STEP, &idx.ssmax_w)?,
            w.f32_buf(STEP, &idx.ssmax_b)?,
            sh.idx_heads,
            sh.idx_d,
            1,
        )?;
        idx.k.project(cx, acts, l, &s.xn, &mut s.kidx)?;
        if rope {
            k.step.enqueue_gather(
                stream,
                &s.kidx,
                &perm.k.0,
                &perm.k.1,
                sh.idx_d,
                &mut s.kidx_moved,
            )?;
        }
        k.index_key.enqueue_index_key_input(
            stream,
            IndexKeyInputArgs {
                x: if rope { &s.kidx_moved } else { &s.kidx },
                gain: &idx.k_norm_w,
                bias: &idx.k_norm_b,
                pos: &s.pos,
                cs: &s.cs,
                eps: sh.idx_eps,
                n_dims: if rope { sh.rope } else { 0 },
                m: 1,
                cache: &mut st.keys,
                fault,
            },
        )?;
        idx.proj.project(cx, acts, l, &s.xn, &mut s.wi)?;
        let mut args = IndexerArgs {
            q: if rope { &s.qi_moved } else { &s.qi },
            w: &s.wi,
            ints: &s.ints,
            n_vis_at: words::N_VIS,
            top_k_at: words::TOP_K,
            tables: &s.table_words,
            rope_at: 0,
            rope_stride: sh.rope,
            keys: &st.keys,
            tokens: 1,
            scratch: &mut st.scratch,
            list: &mut st.list,
            stride: sh.top_k,
        };
        let indexer = if rope {
            &k.indexer_rope
        } else {
            &k.indexer_nope
        };
        indexer.enqueue_score(stream, &mut args)?;
        indexer.enqueue_topk(stream, &mut args, &s.ints, words::N_VIS, words::TOP_K)?;
    }
    // 6. the absorbed query: each head's nope slice dotted into the latent
    //    width, a rope layer's flash row `[q_rope | q_nope2]` assembled and
    //    the whole row scaled by the main SSMax.
    let (qs, dd) = w.q8_planes(STEP, &a.k_b)?;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: &s.q,
            rows_per_head: sh.latent,
            x_head_stride: sh.head_k,
            y_head_stride: width,
            y_off: if rope { sh.rope } else { 0 },
            y: &mut s.q_cur,
        },
    )?;
    if rope {
        k.step.enqueue_gather(
            stream,
            &s.q_rope,
            &perm.f_rope.0,
            &perm.f_rope.1,
            sh.heads * sh.rope,
            &mut s.q_cur,
        )?;
    }
    elem.enqueue_ssmax_rows(
        stream,
        &mut s.q_cur,
        &s.pos,
        w.f32_buf(STEP, &a.ssmax_w)?,
        w.f32_buf(STEP, &a.ssmax_b)?,
        sh.heads,
        width,
        1,
    )?;
    // 7. the attention over the picked rows.
    let list = stores[a.kind.indexer]
        .idx
        .as_ref()
        .map(|st| &st.list)
        .ok_or_else(|| GpuError::State {
            what: STEP,
            missing: "the indexer layer's list",
        })?;
    let cache = stores[a.kind.cache]
        .latent
        .as_ref()
        .ok_or_else(|| GpuError::State {
            what: STEP,
            missing: "the cache layer's latent store",
        })?;
    if rope {
        k.latent.enqueue_kv_gather_sel(
            stream,
            KvGatherSelArgs {
                list,
                n: &s.n_keys,
                stride: sh.top_k,
                src: cache,
                fault,
                dst: &mut s.gathered,
            },
        )?;
        let inputs = bloomery_gpu::flash::FlashInputs {
            q: &s.q_cur,
            kv: &s.gathered,
            // The gathered rows already hold the current row, so the flash
            // counts it on top of this word, not of the cache's key count.
            n_keys_buf: &s.flash_n,
            kq_scale: kq_scale(sh),
            geom: bloomery_gpu::flash::FlashGeom {
                tokens: 1,
                heads: sh.heads,
                rope_dims: sh.rope,
                latent_dims: sh.latent,
            },
        };
        if segments_for(sh.top_k) == 1 {
            k.flash.enqueue_flash_latent(
                stream,
                bloomery_gpu::flash::FlashLatentArgs {
                    inputs,
                    y: &mut s.att,
                },
            )?;
        } else {
            k.flash.enqueue_flash_latent_mma(
                stream,
                FlashSegArgs {
                    inputs,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                },
            )?;
            k.flash.enqueue_flash_merge(
                stream,
                FlashMergeArgs {
                    n_keys_buf: &s.flash_n,
                    cache_rows: sh.top_k,
                    tokens: 1,
                    heads: sh.heads,
                    latent_dims: sh.latent,
                    part_v: &s.part_v,
                    part_ms: &s.part_ms,
                    y: &mut s.att,
                },
            )?;
        }
    } else {
        // SAFETY: a view of no rows at the address of a live u16 allocation
        // aligned for u16 that outlives the view; the walk reads no window
        // row through it, and the view is released below before the cache
        // can drop.
        let window = unsafe {
            DeviceTensor::<u16>::window(cache.buf().cu_deviceptr(), 0, sh.latent, gpu.context())
        };
        let r = k.attn.enqueue(
            stream,
            AttnArgs {
                q: &s.q_cur,
                window: &window,
                compressed: Some(cache),
                selected: Some(SelectedRows {
                    rows: list,
                    stride: sh.top_k,
                }),
                vis: &s.vis,
                sinks: &s.sinks,
                scale: kq_scale(sh),
                tokens: 1,
                heads: sh.heads,
                part_v: &mut s.part_v,
                part_ms: &mut s.part_ms,
                y: &mut s.att,
                fault,
            },
        );
        DeviceTensor::release(window);
        r?;
    }
    // 8. the heads' values up-projected, the output projection.
    let (qs, dd) = w.q8_planes(STEP, &a.v_b)?;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: &s.att,
            rows_per_head: sh.head_v,
            x_head_stride: sh.latent,
            y_head_stride: sh.head_v,
            y_off: 0,
            y: &mut s.av,
        },
    )?;
    a.o.project(cx, acts, l, &s.av, &mut s.r)
}

/// The feed-forward half of layer `l` (stage 9): `s.r` and the residual `x`
/// in, the block's output added to the residual into `x` itself.
#[allow(
    clippy::too_many_arguments,
    reason = "the half's launches: its card and kernels, the shape, the site context, the layer and its number, the routing results, the activation forms, the buffers and the residual (rust-quality R8)"
)]
fn ffn(
    gpu: &Gpu,
    k: &Kernels,
    sh: Shape,
    cx: &GqaCx<'_>,
    (a, l): (&Layer, usize),
    routes: &mut [Option<RouterBufs>],
    acts: &mut Acts,
    (act_hidden, act_ff): (&mut Q8Act, &mut Q8Act),
    s: &mut Scratch,
    x: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let fault = gpu.layer_sink(l)?;
    // The block's input is the residual first — the reference's `ffn_inp` —
    // and the block runs on its norm.
    elem.enqueue_add(stream, x, &s.r, sh.hidden, &mut s.ffn_inp)?;
    elem.enqueue_rms_norm(
        stream,
        &s.ffn_inp,
        cx.w.f32_buf(STEP, &a.ffn_norm)?,
        sh.eps,
        sh.hidden,
        1,
        &mut s.xn,
    )?;
    match &a.ffn {
        Ffn::Dense(d) => {
            d.gate.project(cx, acts, l, &s.xn, &mut s.g)?;
            d.up.project(cx, acts, l, &s.xn, &mut s.u)?;
            elem.enqueue_swiglu(stream, &s.g, &s.u, sh.dense_ff, &mut s.act)?;
            d.down.project(cx, acts, l, &s.act, &mut s.ffn_out)?;
        }
        Ffn::Moe {
            router,
            gate,
            up,
            down,
            ty,
            ty_down,
            shared,
        } => {
            let route = routes[l].as_mut().ok_or_else(|| GpuError::State {
                what: STEP,
                missing: "the layer's routing results",
            })?;
            let w_router = cx.w.f32_tensor(STEP, router)?;
            gpu.q8f32()
                .enqueue_f32_gemv(stream, w_router, &s.xn, 1, &mut route.logits)?;
            k.router.enqueue_route(stream, 1, fault, route)?;
            gpu.enqueue_quantize_q8_1_layer(&s.xn, act_hidden, l)?;
            k.iq.enqueue_gate_up(
                stream,
                &IqGateUp {
                    ty: *ty,
                    wg: stack(cx.w, gate)?,
                    wu: stack(cx.w, up)?,
                    act: act_hidden,
                    sel: &route.ids,
                    n_slots: sh.used,
                    rows_per_expert: sh.ff,
                    slots_per_col: sh.used,
                    rule: GateAct::SiluMul,
                },
                fault,
                &mut s.up,
            )?;
            gpu.enqueue_quantize_q8_1_layer(&s.up, act_ff, l)?;
            k.iq.enqueue_gemv_sel(
                stream,
                *ty_down,
                &SelDown {
                    w: stack(cx.w, down)?,
                    act: act_ff,
                    sel: &route.ids,
                    n_slots: sh.used,
                    rows_per_expert: sh.hidden,
                },
                fault,
                &mut s.rows,
            )?;
            elem.enqueue_weighted_sum(
                stream,
                &s.rows,
                &route.weights,
                sh.hidden,
                sh.used as u32,
                1,
                &mut s.moe,
            )?;
            shared.gate.project(cx, acts, l, &s.xn, &mut s.g)?;
            shared.up.project(cx, acts, l, &s.xn, &mut s.u)?;
            let shared_ff = sh.n_shared * sh.ff;
            elem.enqueue_swiglu(stream, &s.g, &s.u, shared_ff, &mut s.act)?;
            shared.down.project(cx, acts, l, &s.act, &mut s.sh)?;
            elem.enqueue_add(stream, &s.moe, &s.sh, sh.hidden, &mut s.ffn_out)?;
        }
    }
    // The block's output back into the running row.
    elem.enqueue_add(stream, &s.ffn_inp, &s.ffn_out, sh.hidden, x)
}

impl LayerTap {
    /// The tap's buffers, sized by the layer's kind (a rope layer's flash
    /// row is the cache's width, a NoPE layer's the latent alone; a shared
    /// layer keeps no indexer taps).
    fn new(stream: &CudaStream, sh: &Shape, kind: Kind) -> Result<LayerTap, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let width = sh.latent + sh.pe(kind.rope);
        let (idx_q, idx_d) = (sh.idx_heads * sh.idx_d, sh.idx_d);
        Ok(LayerTap {
            attn_norm: z(sh.hidden)?,
            q_cur: z(sh.heads * width)?,
            ffn_inp: z(sh.hidden)?,
            l_out: z(sh.hidden)?,
            kv_a: z(width)?,
            kv_s: z(width)?,
            kvr: z(width)?,
            qi: z(idx_q)?,
            kidx: z(idx_d)?,
            wi: z(sh.idx_heads)?,
            q: z(sh.heads * sh.head_k)?,
            q_rope: z(sh.heads * sh.head_k)?,
            list: DeviceBuffer::zeroed(stream, sh.top_k)?,
            cache16: DeviceBuffer::zeroed(stream, sh.ctx * width)?,
            att: z(sh.heads * sh.latent)?,
            av: z(sh.heads * sh.head_v)?,
            moe: z(sh.hidden)?,
            sh: z(sh.hidden)?,
            ffn_out: z(sh.hidden)?,
            rope: kind.rope,
        })
    }

    /// The buffers' bytes.
    fn bytes(&self) -> usize {
        [
            self.attn_norm.num_bytes(),
            self.q_cur.num_bytes(),
            self.ffn_inp.num_bytes(),
            self.l_out.num_bytes(),
            self.kv_a.num_bytes(),
            self.kv_s.num_bytes(),
            self.kvr.num_bytes(),
            self.qi.num_bytes(),
            self.kidx.num_bytes(),
            self.wi.num_bytes(),
            self.q.num_bytes(),
            self.q_rope.num_bytes(),
            self.list.num_bytes(),
            self.cache16.num_bytes(),
            self.att.num_bytes(),
            self.av.num_bytes(),
            self.moe.num_bytes(),
            self.sh.num_bytes(),
            self.ffn_out.num_bytes(),
        ]
        .into_iter()
        .sum()
    }

    /// Copy the layer's compared values as the layer leaves them (the
    /// module doc's): the scaled query, the two residuals, the key path's
    /// three forms, the indexer's queries, key and weights and the block's
    /// outputs, beside the list the layer read. Pure copies, ordered on the
    /// engine stream; the attention's own norm row is
    /// [`LayerTap::snap_attention`]'s.
    fn snap(
        &mut self,
        stream: &CudaStream,
        s: &Scratch,
        x: &DeviceBuffer<f32>,
        list: Option<&DeviceBuffer<u32>>,
        cache: Option<&DeviceTensor<u16>>,
        sh: Shape,
    ) -> Result<(), GpuError> {
        let (q_cur, kv, kv_s, kvr, qi, kidx, wi) = (
            self.q_cur.len(),
            self.kv_a.len(),
            self.kv_s.len(),
            self.kvr.len(),
            self.qi.len(),
            self.kidx.len(),
            self.wi.len(),
        );
        copy_row(stream, &s.q_cur, &mut self.q_cur, q_cur)?;
        copy_row(stream, &s.ffn_inp, &mut self.ffn_inp, sh.hidden)?;
        copy_row(stream, x, &mut self.l_out, sh.hidden)?;
        copy_row(stream, &s.kv_a, &mut self.kv_a, kv)?;
        copy_row(stream, &s.kv_s, &mut self.kv_s, kv_s)?;
        copy_row(stream, &s.kvr, &mut self.kvr, kvr)?;
        copy_row(stream, &s.q, &mut self.q, sh.heads * sh.head_k)?;
        copy_row(stream, &s.q_rope, &mut self.q_rope, sh.heads * sh.head_k)?;
        let (qi_at, kidx_at) = if self.rope {
            (&s.qi_moved, &s.kidx_moved)
        } else {
            (&s.qi, &s.kidx)
        };
        copy_row(stream, qi_at, &mut self.qi, qi)?;
        copy_row(stream, kidx_at, &mut self.kidx, kidx)?;
        copy_row(stream, &s.wi, &mut self.wi, wi)?;
        copy_row(stream, &s.moe, &mut self.moe, sh.hidden)?;
        copy_row(stream, &s.sh, &mut self.sh, sh.hidden)?;
        copy_row(stream, &s.ffn_out, &mut self.ffn_out, sh.hidden)?;
        if let Some(list) = list {
            self.list.copy_from_device_async(list, stream)?;
        }
        if let Some(cache) = cache {
            self.cache16.copy_from_device_async(cache.buf(), stream)?;
        }
        copy_row(stream, &s.att, &mut self.att, sh.heads * sh.latent)?;
        copy_row(stream, &s.av, &mut self.av, sh.heads * sh.head_v)?;
        Ok(())
    }

    /// The attention block's own norm row, copied between the attention and
    /// the ffn: the ffn's `rms_norm` writes the same scratch.
    fn snap_attention(
        &mut self,
        stream: &CudaStream,
        s: &Scratch,
        sh: Shape,
    ) -> Result<(), GpuError> {
        copy_row(stream, &s.xn, &mut self.attn_norm, sh.hidden)
    }

    /// The tap's values, the scratch copies and the list the layer read (the
    /// routing is read where it lives, at [`Body::nodes`]); it synchronizes
    /// the stream.
    fn to_host(&self, stream: &CudaStream) -> Result<LayerNodes, GpuError> {
        let v = |b: &DeviceBuffer<f32>| b.to_host_vec(stream);
        Ok(LayerNodes {
            attn_norm: v(&self.attn_norm)?,
            q_cur: v(&self.q_cur)?,
            ffn_inp: v(&self.ffn_inp)?,
            l_out: v(&self.l_out)?,
            kv_a: v(&self.kv_a)?,
            kv_s: v(&self.kv_s)?,
            kvr: v(&self.kvr)?,
            qi: v(&self.qi)?,
            kidx: v(&self.kidx)?,
            wi: v(&self.wi)?,
            q: v(&self.q)?,
            q_rope: v(&self.q_rope)?,
            list: self.list.to_host_vec(stream)?,
            cache16: self.cache16.to_host_vec(stream)?,
            att: v(&self.att)?,
            av: v(&self.av)?,
            moe: v(&self.moe)?,
            sh: v(&self.sh)?,
            ffn_out: v(&self.ffn_out)?,
            route_logits: Vec::new(),
            route_ids: Vec::new(),
            route_probs: Vec::new(),
            route_weights: Vec::new(),
        })
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = bloomery_gpu::model::NoHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> model::arch::Arch {
        model::arch::Arch::MistralLarge4
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step's words, each in one copy: the position, the token, the
    /// walk's visible pair, the selected attention's count and the rope row
    /// (its f32 values and their bits for the score pass's table).
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::Body::refresh";
        let DecodeInput { token, pos } = *input;
        let sh = self.sh;
        if pos as usize >= sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("position {pos} past the cache's {} rows", sh.ctx),
            ));
        }
        let n = (pos + 1) as usize;
        self.s.host_pos[words::POS] = pos;
        // `n_vis` (word 0) is the embedding launch's own write; the width
        // beside it never moves, so the host copy carries it every step.
        self.s.host_ints[words::TOP_K] = sh.top_k as u32;
        self.s.host_token[0] = token;
        self.s.host_vis = vec![0, n.min(sh.top_k) as u32];
        self.s.host_n_keys[words::N_KEYS] = n.min(sh.top_k) as u32;
        // The gathered flash's rows are the selection's own count
        // (`min(pos + 1, top_k)`): the flash counts the current row on top
        // of the word it reads and drops it again for the one token
        // (`m = 1`), so the word names the whole visible count.
        self.s.host_flash_n[0] = (pos + 1).min(sh.top_k as u32);
        self.s.host_cs.clear();
        self.table
            .push(pos, Direction::Forward, &mut self.s.host_cs);
        self.s.host_table_words = self.s.host_cs.iter().map(|v| v.to_bits()).collect();
        self.s.pos.copy_from_host(stream, &self.s.host_pos)?;
        self.s.token.copy_from_host(stream, &self.s.host_token)?;
        // `n_vis` the embedding launch rewrites; the width beside it stays.
        self.s.ints.copy_from_host(stream, &self.s.host_ints)?;
        self.s.vis.copy_from_host(stream, &self.s.host_vis)?;
        self.s.n_keys.copy_from_host(stream, &self.s.host_n_keys)?;
        self.s
            .flash_n
            .copy_from_host(stream, &self.s.host_flash_n)?;
        self.s.cs.copy_from_host(stream, &self.s.host_cs)?;
        self.s
            .table_words
            .copy_from_host(stream, &self.s.host_table_words)?;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue(gpu, w, head)
    }

    /// Rewind every store to empty: each store is a plane of positions the
    /// launches read only below the position's own live count, and every
    /// step writes its position's row before it reads it, so nothing a
    /// later call reads survives a cut. The routers' pad lanes are the
    /// load's own and no launch writes them.
    fn reset(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Rms
    }

    fn resident_bytes(&self) -> usize {
        let stores: usize = self
            .stores
            .iter()
            .map(|st| {
                st.latent.as_ref().map_or(0, |c| c.buf().num_bytes())
                    + st.idx.as_ref().map_or(0, |i| {
                        i.keys.buf().num_bytes() + i.list.num_bytes() + i.scratch.device_bytes()
                    })
            })
            .sum();
        stores
            + self
                .routes
                .iter()
                .map(|r| r.as_ref().map_or(0, RouterBufs::bytes))
                .sum::<usize>()
            + self.acts.bytes()
            + self.act_hidden.device_bytes()
            + self.act_ff.device_bytes()
            + self.s.bytes()
            + self
                .taps
                .as_ref()
                .map_or(0, |t| t.layers.iter().map(|a| a.bytes()).sum::<usize>())
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Self::Host> {
        None
    }
}

impl bloomery_gpu::model::Instrumented for Body {
    /// Fill the first `rows` cache rows of every store with the body's own
    /// synthetic pattern — the latent rows a position-scaled ramp, the
    /// index-key rows a sign-alternating one, both inside the f16 normals
    /// and the rows past them zero: the state a prompt of `rows` tokens
    /// could have left behind, not what the model would have written.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::Body::seed_depth";
        if rows > self.sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("{rows} rows past the cache's {}", self.sh.ctx),
            ));
        }
        let stream = gpu.stream();
        let ramp = |(r, w): (usize, usize)| -> u16 {
            if r >= rows {
                return 0;
            }
            let v = if r % 2 == 0 {
                0.125 + 0.001 * (r % 97) as f32
            } else {
                0.0625 + 0.0005 * ((r + w) % 31) as f32
            };
            gguf::quant::f32_to_f16_bits(v)
        };
        for st in &mut self.stores {
            if let Some(latent) = st.latent.as_mut() {
                let (r, w) = (latent.rows(), latent.cols());
                let host: Vec<u16> = (0..r * w).map(|i| ramp((i / w, i % w))).collect();
                *latent = DeviceTensor::upload(stream, &host, r, w)?;
            }
            if let Some(idx) = st.idx.as_mut() {
                let (r, w) = (idx.keys.rows(), idx.keys.cols());
                let host: Vec<u16> = (0..r * w)
                    .map(|i| {
                        let (row, col) = (i / w, i % w);
                        if row >= rows {
                            return 0;
                        }
                        let sign = if ((row + col) / 8) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        gguf::quant::f32_to_f16_bits(sign * (0.25 + 0.001 * (col % 13) as f32))
                    })
                    .collect();
                idx.keys = DeviceTensor::upload(stream, &host, r, w)?;
            }
        }
        Ok(())
    }
}

impl Scratch {
    /// Device bytes the buffers hold.
    fn bytes(&self) -> usize {
        [
            self.xn.num_bytes(),
            self.x_ln.num_bytes(),
            self.q.num_bytes(),
            self.q_rope.num_bytes(),
            self.q_cur.num_bytes(),
            self.kv_a.num_bytes(),
            self.kv_s.num_bytes(),
            self.kvr.num_bytes(),
            self.qi.num_bytes(),
            self.qi_moved.num_bytes(),
            self.wi.num_bytes(),
            self.kidx.num_bytes(),
            self.kidx_moved.num_bytes(),
            self.att.num_bytes(),
            self.av.num_bytes(),
            self.r.num_bytes(),
            self.ffn_inp.num_bytes(),
            self.g.num_bytes(),
            self.u.num_bytes(),
            self.act.num_bytes(),
            self.up.num_bytes(),
            self.rows.num_bytes(),
            self.moe.num_bytes(),
            self.sh.num_bytes(),
            self.ffn_out.num_bytes(),
            self.gathered.buf().num_bytes(),
            self.pos.num_bytes(),
            self.token.num_bytes(),
            self.pos_out.num_bytes(),
            self.ints.num_bytes(),
            self.vis.num_bytes(),
            self.n_keys.num_bytes(),
            self.cs.num_bytes(),
            self.table_words.num_bytes(),
            self.part_v.num_bytes(),
            self.part_ms.num_bytes(),
            self.sinks.num_bytes(),
        ]
        .into_iter()
        .sum()
    }
}

impl Body {
    /// Arm or disarm the per-layer taps (the module doc's copies). The
    /// caller holds the graph rule: a capture that exists would not hold
    /// the copies, so its holder refuses before calling.
    pub fn arm_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        if !on {
            self.taps = None;
            return Ok(());
        }
        let stream = gpu.stream();
        let sh = self.sh;
        let mut layers = Vec::with_capacity(sh.n_layer);
        for a in &self.layers {
            layers.push(LayerTap::new(stream, &sh, a.kind)?);
        }
        self.taps = Some(Taps {
            layers,
            inject: Vec::new(),
            span: None,
        });
        Ok(())
    }

    /// Hold the chain to the layers of `span` (and the head only when the
    /// span ends at the model's end), `None` the whole chain. The caller
    /// holds the graph rule: a captured step replays the walk it recorded.
    pub fn set_layer_span(&mut self, span: Option<Range<usize>>) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::set_layer_span";
        let want = span.filter(|s| s.start < s.end);
        if let Some(s) = &want
            && (s.start >= self.layers.len() || s.end > self.layers.len())
        {
            return Err(GpuError::shape(
                WHAT,
                format!("the span {s:?} is past the model's {}", self.layers.len()),
            ));
        }
        match &mut self.taps {
            Some(t) => t.span = want,
            None => {
                return Err(GpuError::State {
                    what: WHAT,
                    missing: "the per-layer taps (arm_taps): the span is their own instrument",
                });
            }
        }
        Ok(())
    }

    /// Replace the running row ahead of layer `at` with `row` (the oracle's
    /// own `l_out` of the layer before it), the instrument the per-layer node
    /// clauses run one layer at a time on. One row a layer: a second call at
    /// `at` replaces its row, a call at another layer of the span adds one —
    /// a span from the layer a list or cache lives on to the layer that
    /// reads it feeds both. The caller holds the graph rule.
    pub fn set_input_override(
        &mut self,
        gpu: &Gpu,
        at: usize,
        row: &[f32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::set_input_override";
        let hidden = self.sh.hidden;
        if at >= self.layers.len() || row.len() != hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {at} of {}, a row of {} values ({} given)",
                    self.layers.len(),
                    hidden,
                    row.len()
                ),
            ));
        }
        let stream = gpu.stream();
        match &mut self.taps {
            Some(t) => {
                let held = t.span.as_ref().is_none_or(|s| s.contains(&at));
                if !held {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("layer {at} outside the span the taps hold"),
                    ));
                }
                match t.inject.iter_mut().find(|(l, _)| *l == at) {
                    Some((_, held)) => held.copy_from_host(stream, row)?,
                    None => t.inject.push((at, DeviceBuffer::from_host(stream, row)?)),
                }
                Ok(())
            }
            None => Err(GpuError::State {
                what: WHAT,
                missing: "the per-layer taps (arm_taps): the override is their own instrument",
            }),
        }
    }

    /// The armed taps' values of `layers`, read back (a gate's instrument;
    /// it synchronizes the stream), beside each layer's routing. Refused by
    /// name: a range past the model's layers.
    pub fn nodes(&mut self, gpu: &Gpu, layers: Range<usize>) -> Result<Vec<LayerNodes>, GpuError> {
        const WHAT: &str = "mistrallarge4::nodes";
        let stream = gpu.stream();
        let n = self.layers.len();
        if layers.start > layers.end || layers.end > n {
            return Err(GpuError::shape(
                WHAT,
                format!("the layers {layers:?} are past the model's {n}"),
            ));
        }
        let taps = self.taps.as_ref().ok_or_else(|| GpuError::State {
            what: WHAT,
            missing: "the per-layer taps (arm_taps)",
        })?;
        let mut out = Vec::with_capacity(layers.len());
        for l in layers {
            let t = taps.layers[l].to_host(stream)?;
            let route = self.routes[l].as_ref();
            let (logits, ids, probs, weights) = match route {
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                Some(r) => (
                    r.logits.to_host_vec(stream)?,
                    r.ids.to_host_vec(stream)?,
                    r.probs.to_host_vec(stream)?,
                    r.weights.to_host_vec(stream)?,
                ),
            };
            out.push(LayerNodes {
                route_logits: logits,
                route_ids: ids,
                route_probs: probs,
                route_weights: weights,
                ..t
            });
        }
        Ok(out)
    }

    pub fn index_scores(&mut self, gpu: &Gpu, l: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT: &str = "mistrallarge4::index_scores";
        let st = self
            .stores
            .get(l)
            .and_then(|s| s.idx.as_ref())
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("layer {l} is not a full indexer of the model"),
                )
            })?;
        Ok(st.scratch.scores.to_host_vec(gpu.stream())?)
    }

    /// The list every full indexer layer of `layers` kept, and the layer it
    /// read (itself on a full layer, its source on a shared one), in layer
    /// order (a gate's instrument; it synchronizes the stream).
    pub fn lists(
        &mut self,
        gpu: &Gpu,
        layers: Range<usize>,
    ) -> Result<Vec<(usize, Vec<u32>)>, GpuError> {
        const WHAT: &str = "mistrallarge4::lists";
        let stream = gpu.stream();
        let n = self.layers.len();
        if layers.start > layers.end || layers.end > n {
            return Err(GpuError::shape(
                WHAT,
                format!("the layers {layers:?} are past the model's {n}"),
            ));
        }
        let mut out = Vec::with_capacity(layers.len());
        for l in layers {
            let src = self.layers[l].kind.indexer;
            let list = &self.stores[src].idx.as_ref().unwrap().list;
            out.push((src, list.to_host_vec(stream)?));
        }
        Ok(out)
    }

    /// Drop every input override ([`Body::set_input_override`]): the chain's
    /// embedding writes the running row again.
    pub fn clear_input_override(&mut self) {
        if let Some(t) = self.taps.as_mut() {
            t.inject.clear();
        }
    }

    /// One routing launch of layer `layer`'s own buffers over a planted
    /// weight whose expert 0 row is NaN and the rest finite and strictly
    /// descending, at a one-hot input: a NaN lane must raise
    /// [`FaultSite::Router`] on the fault word and be picked by no slot
    /// (a gate's instrument; it synchronizes the stream).
    pub fn router_lane_nan(&mut self, gpu: &Gpu, layer: usize) -> Result<(bool, bool), GpuError> {
        const WHAT: &str = "mistrallarge4::router_lane_nan";
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let (hidden, n_expert, used) = (self.sh.hidden, self.sh.experts, self.sh.used);
        let route = self
            .routes
            .get_mut(layer)
            .and_then(|r| r.as_mut())
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} is not a routed one")))?;
        let mut w = vec![0.0f32; n_expert * hidden];
        w[..hidden].fill(f32::NAN);
        for (r, logit) in w.chunks_mut(hidden).enumerate().skip(1) {
            logit[0] = 20.0 - r as f32;
        }
        let mut x = vec![0.0f32; hidden];
        x[0] = 1.0;
        let w_dev = DeviceTensor::upload(stream, &w, n_expert, hidden)?;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        gpu.q8f32()
            .enqueue_f32_gemv(stream, &w_dev, &x_dev, 1, &mut route.logits)?;
        self.k
            .router
            .enqueue_route(stream, 1, gpu.layer_sink(layer)?, route)?;
        let ids = route.ids.to_host_vec(stream)?;
        let weights = route.weights.to_host_vec(stream)?;
        let raised = gpu
            .take_fault()?
            .is_some_and(|f| f.code == FaultSite::Router as u32);
        let defined = ids.len() == used
            && ids
                .iter()
                .all(|&i| usize::try_from(i).is_ok_and(|i| i > 0 && i < n_expert))
            && ids.iter().collect::<std::collections::BTreeSet<_>>().len() == ids.len()
            && weights.iter().all(|v| v.is_finite());
        Ok((raised, defined))
    }
}
