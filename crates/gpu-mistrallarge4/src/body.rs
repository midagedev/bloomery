//! The Mistral Large 4 placed body: one [`Body`] behind [`ChainBody`], the
//! plan's tensors on one stage card and the routed experts the card does not
//! hold on the host tier, with the plan's expert tier card beside the stage
//! card holding each layer's next ids ([`Body::open_placed`]), the step one
//! token's walk over the layers (`crate::step`). The launch order is the
//! reference graph's (`src/models/mistral-large4.cpp`):
//!
//! 1. the embedding row of the token, read from the file's table on the host
//!    (the plan keeps the table off the card) and copied into the running row
//!    before the step;
//! 2. `attn_norm`, with its q8_1 form in the same launch (`norm_quant`) where
//!    `attn_q` reads the q8_1 rows, else alone (`rms_norm`);
//! 3. `attn_q` of the normed row — a head's `[nope | rope]` values —
//!    and the rope of every 64-value column of it (`rope`; a rotated nope
//!    column is never read);
//! 4. on a layer that owns its cache, `attn_kv_a_mqa` (Q8_0) of the normed
//!    row, then the fused key path (`kv_norm_rope_append`): the latent's RMS
//!    norm, the rope of the row's tail, the `[k_rope | kv_compressed]` f16
//!    row appended at the position — a NoPE owner appends its bare latent
//!    (`latent_append`), and a reader layer appends nothing;
//! 5. on a full-indexer layer, the lightning indexer: the LayerNorm of the
//!    normed row (`ln_rows`) and `indexer.attn_q_b` of it, the file's
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
//! 8. `attn_v_b` up-projects each head's latent output (`q8_0_gemv_heads` over
//!    Q8_0 planes, else the down `_sel` entry of the file's type with the
//!    heads as the experts), `attn_output` projects the heads' values, and
//!    the residual add makes `ffn_inp`;
//! 9. the feed-forward half through the common routed block
//!    ([`crate::ffn::Routed`]): the dense lead's `ffn_norm` and gate·up·down
//!    with its residual add; on a routed layer the front (`ffn_norm` into the
//!    host boundary, the softmax top-4 router over the file's 128 experts in
//!    one launch, the handoff and the go), the shadow under the host leg (the
//!    shared expert and the card's slots of the picked experts), then the
//!    back (the wait and one join: the host's sum plus the card's, plus the
//!    shared output, plus `ffn_inp`, mainline's association); on a layer the
//!    expert tier card holds experts of, the card's sum is over the stage
//!    card's slots and the tier's in slot order, the sum a card holding both
//!    would make;
//! 10. the head: the final RMS norm, the `output` projection of the file's
//!     type, the tanh softcap of the file's bound and the argmax
//!     (`Head::enqueue_with_epilogue`).
//!
//! Every count the chain reads — the position, the live rows, the list's
//! width, the walk's visible pair, the rope row — lives in one of five small
//! device buffers the host refreshes before a replay, beside the running
//! row, so one capture serves every position.
//!
//! The taps (`arm_taps`) hold a copy of every compared value a layer leaves,
//! read back by a gate; they refuse a captured graph, which would not hold
//! the copies. A routed layer's `ffn_moe_out` exists in no buffer of the
//! step (the join sums it in its launch): its tap adds the join's own terms
//! ([`crate::ffn::JoinTerms`]) in the join's order. A tier layer's card sum
//! is made inside the join over both cards' slots, so a load with an expert
//! tier card refuses the taps by name.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::{Acts, DecodeInput, GqaCx};
use bloomery_gpu::flash::{FlashKernels, FlashMergeArgs, FlashSegArgs, segments_for};
use bloomery_gpu::flash_gqa::FlashGqaKernels;
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::head::{Head, HeadNorm};
use bloomery_gpu::host::leg_tier::{Hang, LegOpen, TierSide};
use bloomery_gpu::host::refuse_tier_count;
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::ResidencyGlue;
use bloomery_gpu::host::tier::TierOpen;
use bloomery_gpu::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use bloomery_gpu::kquant::SelDown;
use bloomery_gpu::latent::{KvGatherSelArgs, LatentAppendArgs, LatentKernels, Rows as LatentRows};
use bloomery_gpu::model::{
    CardBytes, ChainBody, GpuModel, HostServed, Q8_0GemvHeadsArgs, StepKernels,
    hosted_hybrid_tiered, refuse_scratch_past, stage_run,
};
use bloomery_gpu::q6k_sel::Q6kSelKernels;
use bloomery_gpu::quant_sel::QuantSelKernels;
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::route_softmax::RouterOut;
use bloomery_gpu::routed_block::{RouteIn, RouterEntry};
use bloomery_gpu::site::{Form, KGemvKernels};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, FaultSink, FaultSite, Gpu, GpuError, Q8Act, copy_row};
use bloomery_gpu_deepseek41::attn::{AttnArgs, AttnKernels, SelectedRows, segments};
use bloomery_gpu_deepseek41::index_key::{IndexKeyInputArgs, IndexKeyKernels};
use bloomery_gpu_deepseek41::indexer::{IndexerArgs, IndexerKernels, IndexerScratch};
use bloomery_gpu_deepseek41::prompt_walk::{self, BatchBufs};
use bloomery_gpu_deepseek41::transpose::TransposeKernels;
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::mistrallarge4::host;
use model::arch::mistrallarge4::hparams::Hparams;
use model::arch::mistrallarge4::place::{IDX_CHUNK, IDX_STEP_ROWS, PROMPT_ROWS, PlanInputs};
use model::arch::mistrallarge4::sites::SelUp;
use model::arch::names;
use model::embed::EmbedTable;
use model::placement::Plan;
use runtime::layer::{Layer as LayerKind, hosted};

use crate::features::FeatureTap;
use crate::ffn::{self, Routed, RoutedDims, Router};
use crate::prefill::{Prefill, T_MAX};
use crate::resid;
use crate::tensors::{self, Kind, Layer, VUp};
use crate::tier;

/// The sequence a server saves, parks and cuts, and the media rows a prompt takes
/// (`seat.rs`, a child module for the stores it reads).
#[path = "seat.rs"]
pub mod seat;

/// The verify of several rows behind `Rows` (`rows.rs`, a child module for
/// the step's buffers it binds per row).
#[path = "rows.rs"]
pub mod rows;

pub(crate) const WHAT: &str = "mistrallarge4::Body::load";
pub(crate) const STEP: &str = "mistrallarge4::Body::enqueue_chain";

/// The placed model.
pub type MistralLarge4Model = GpuModel<Body>;

/// The step's word buffers, refreshed before a replay: the position (read at
/// word 0 by every launch that takes a position), the indexer's counts (the
/// live rows and the list's width), the walk's visible pair and the selected
/// attention's count.
mod words {
    /// The position, word 0 of its own buffer.
    pub(crate) const POS: usize = 0;
    /// `ints[0]`: the live rows, `pos + 1`.
    pub(crate) const N_VIS: usize = 0;
    /// `ints[1]`: the list's width, written once at load.
    pub(crate) const TOP_K: usize = 1;
    /// The selected attention's count, word 0 of its own buffer.
    pub(crate) const N_KEYS: usize = 0;
}

/// The kernels the step launches beyond the `Gpu`'s own and the routed
/// block's.
pub(crate) struct Kernels {
    pub(crate) fused: FusedKernels,
    pub(crate) latent: LatentKernels,
    pub(crate) flash: FlashKernels,
    pub(crate) attn: AttnKernels,
    pub(crate) index_key: IndexKeyKernels,
    /// The lightning indexer at the file's two rope widths: a roped layer's
    /// `rope.dimension_count`, and 0 on a NoPE layer's index head.
    pub(crate) indexer_rope: IndexerKernels,
    pub(crate) indexer_nope: IndexerKernels,
    /// The prompt batch's layout change before an indexer launch: its rows' index queries and
    /// head weights, written a row a token, into the score pass's token per column.
    pub(crate) transpose: TransposeKernels,
    iq: QuantSelKernels,
    step: StepKernels,
    pub(crate) kgemv: KGemvKernels,
    pub(crate) g32: Gemm32Kernels,
    pub(crate) flash_gqa: FlashGqaKernels,
    /// The Q6_K down `_sel`, the heads' up-projection of a Q6_K `attn_v_b`.
    q6: Q6kSelKernels,
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
            transpose: TransposeKernels::load(ctx)?,
            iq: QuantSelKernels::load(ctx, gpu.fault_word())?,
            step: StepKernels::load(ctx)?,
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash_gqa: FlashGqaKernels::load(ctx)?,
            q6: Q6kSelKernels::load(ctx, gpu.fault_word())?,
        })
    }
}

/// The widths the step's buffers are cut for, from the file's own keys.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    /// `attention.key_length_mla`: a query head's whole width.
    pub(crate) head_k: usize,
    pub(crate) head_v: usize,
    /// `attention.kv_lora_rank`: the cached latent's width.
    pub(crate) latent: usize,
    /// `rope.dimension_count`.
    pub(crate) rope: usize,
    /// `attention.indexer.head_count` and `key_length`: the lightning
    /// indexer's own heads of values.
    pub(crate) idx_heads: usize,
    pub(crate) idx_d: usize,
    /// `attention.indexer.top_k`: the list's width.
    pub(crate) top_k: usize,
    pub(crate) experts: usize,
    pub(crate) used: usize,
    pub(crate) ff: usize,
    pub(crate) eps: f32,
    /// The indexer's LayerNorms' own epsilon.
    pub(crate) idx_eps: f32,
    pub(crate) ctx: usize,
    pub(crate) n_layer: usize,
    pub(crate) softcap: f32,
}

/// A full indexer layer's stores.
pub(crate) struct IdxStore {
    pub(crate) keys: DeviceTensor<u16>,
    list: DeviceBuffer<u32>,
    scratch: IndexerScratch,
}

/// The step's own buffers, allocated once at load.
pub(crate) struct Scratch {
    /// The attention's normed row, and the dense lead's.
    pub(crate) xn: DeviceBuffer<f32>,
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
    /// The heads' latent outputs in the q8_1 form a down `_sel` v_b reads,
    /// one column a head, and the ids that make each head its own expert.
    v_act: Q8Act,
    v_ids: DeviceBuffer<u32>,
    av: DeviceBuffer<f32>,
    /// The attention's output projection, the residual's delta.
    pub(crate) r: DeviceBuffer<f32>,
    /// The attention residual (mainline's `ffn_inp`), the routed block's
    /// input.
    pub(crate) ffn_inp: DeviceBuffer<f32>,
    /// The dense lead's block output.
    pub(crate) ffn_out: DeviceBuffer<f32>,
    /// The rope layer's gathered cache rows, `top_k` of the cache's width.
    gathered: DeviceTensor<u16>,
    pos: DeviceBuffer<u32>,
    ints: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// The gathered flash's live row count, `min(pos + 1, top_k)`.
    /// With one query token, `causal_limit` reads exactly this many rows.
    flash_n: DeviceBuffer<u32>,
    cs: DeviceBuffer<f32>,
    /// The rope row's f32 bits as u32 words, the score pass's own table.
    table_words: DeviceBuffer<u32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    host_pos: Vec<u32>,
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
pub(crate) struct Perm {
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
    /// The routing of a routed layer: the experts' logits, the kept ids,
    /// their probabilities and their renormalized weights.
    pub route_logits: Vec<f32>,
    pub route_ids: Vec<u32>,
    pub route_probs: Vec<f32>,
    pub route_weights: Vec<f32>,
}

/// The body: the shape, each layer's tensors and stores, the kernels, the
/// activation forms, the step's buffers, the routed blocks and the host tier.
pub struct Body {
    pub(crate) sh: Shape,
    pub(crate) layers: Vec<Layer>,
    /// Each layer's kind as the walk reads it: a dense lead runs whole in its
    /// front, a routed layer has a host leg.
    pub(crate) cfg: Vec<LayerKind>,
    pub(crate) stores: Vec<LayerStore>,
    pub(crate) k: Kernels,
    pub(crate) acts: Acts,
    pub(crate) perm: Perm,
    pub(crate) s: Scratch,
    /// The running row: the embedding, then each layer's output but the
    /// last's, which goes to the head's input.
    pub(crate) x: DeviceBuffer<f32>,
    pub(crate) table: RopeTable,
    pub(crate) taps: Option<Taps>,
    /// The list a gate forces into a layer's selection at the last position of a call, until it
    /// clears it.
    pub(crate) forced: Option<Forced>,
    /// The rows a block draft reads, when its tap is attached
    /// ([`crate::features::attach_features`]).
    pub(crate) feat: Option<FeatureTap>,
    pub(crate) ffn: Routed,
    pub(crate) shell: TierShell<HostRun>,
    /// The slot map's card copy the routed blocks and the residency read.
    slots: Arc<DeviceTensor<u32>>,
    /// The token embedding, read on the host, and the step's row of it.
    pub(crate) embed: EmbedTable,
    row: Vec<f32>,
    /// What the served body carries beside the step ([`seat::State`]).
    seat: seat::State,
    /// The buffers of a verify's rows past row 0, bound in row 0's place for
    /// their parts of the walk ([`rows::RowSet`]).
    pub(crate) pass_rows: rows::RowSet,
    /// The prompt batch's units of a group (the common walk's) and its own buffers
    /// ([`crate::prefill`]), within the plan's prompt reserve.
    pub(crate) units: BatchBufs,
    pub(crate) prefill: Prefill,
    pub(crate) prompt_reserve: u64,
}

/// A layer's stores: its latent cache (an owner's), the index-key cache, the
/// list and the scratch (a full indexer's own).
pub(crate) struct LayerStore {
    /// `[ctx × (latent + pe)]` f16, `None` on a reader layer.
    pub(crate) latent: Option<DeviceTensor<u16>>,
    pub(crate) idx: Option<IdxStore>,
}

impl LayerStore {
    /// The layer's list, on a full indexer layer.
    pub(crate) fn list(&self) -> Option<&DeviceBuffer<u32>> {
        self.idx.as_ref().map(|i| &i.list)
    }

    /// The layer's latent cache, on a layer that owns one.
    pub(crate) fn latent(&self) -> Option<&DeviceTensor<u16>> {
        self.latent.as_ref()
    }

    /// The K/V planes the plan counts: the latent rows and the index keys.
    fn kv_bytes(&self) -> usize {
        self.latent.as_ref().map_or(0, |c| c.buf().num_bytes())
            + self.idx.as_ref().map_or(0, |i| i.keys.buf().num_bytes())
    }

    /// The rest, the step's scratch: the list and the indexer's scores.
    fn scratch_bytes(&self) -> usize {
        self.idx
            .as_ref()
            .map_or(0, |i| i.list.num_bytes() + i.scratch.device_bytes())
    }

    /// The indexer's scores over the context, a full layer's own: the part of the scratch the plan
    /// counts by position.
    fn scores_bytes(&self) -> usize {
        self.idx
            .as_ref()
            .map_or(0, |i| i.scratch.scores.num_bytes())
    }
}

/// The per-layer staging a gate's taps read, one a layer, and the router the
/// taps rerun for a routed layer's logits and probabilities (the step's
/// launch over the step's normed row: the same bits).
pub(crate) struct Taps {
    pub(crate) layers: Vec<LayerTap>,
    /// The rows that replace the running row ahead of their layer, at most
    /// one a layer.
    pub(crate) inject: Vec<(usize, DeviceBuffer<f32>)>,
    pub(crate) span: Option<Range<usize>>,
    pub(crate) router: Router,
    pub(crate) route: RouterOut,
}

impl Taps {
    /// Whether the chain runs layer `l`: every layer but those outside the
    /// span.
    pub(crate) fn runs(&self, l: usize) -> bool {
        self.span.as_ref().is_none_or(|s| s.contains(&l))
    }
}

/// A list a gate forces into a full layer's selection ([`Body::force_selection`]): the layer
/// and the `top_k` rows that stand for its pick at a call's last position.
pub(crate) struct Forced {
    pub(crate) layer: usize,
    pub(crate) list: DeviceBuffer<u32>,
}

/// One layer's compared values, copied as the layer leaves them.
pub(crate) struct LayerTap {
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
    /// A routed layer's routed sum and shared output, made from the join's
    /// terms; the block's output (the dense lead's own).
    moe: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    /// A routed layer's routing: the logits, the probabilities, the kept ids
    /// and their weights; `None` on the dense lead.
    route: Option<RouteTap>,
    /// The layer ropes: its index taps copy the moved rows, a NoPE layer's
    /// the projections as written.
    rope: bool,
}

/// A routed layer's copy of its routing.
struct RouteTap {
    logits: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl Body {
    /// The placed load of `file` by `plan` on its card `card`: the plan's
    /// segments resident on that card, the routed experts it keeps on the host
    /// served by the host tier under `host`, the stores of the plan's context
    /// and the step's buffers. One stage card runs every layer and the head;
    /// the plan's expert tier card, when it names one, holds each routed
    /// layer's ids after the stage's prefix and is hung under the host tier
    /// (the tier layers' join is the common routed block's). Both files of
    /// the family load through this one path.
    ///
    /// # Errors
    /// A plan of other than one stage card, more expert tier cards than the
    /// host tier serves, a context past the trained one, a head, key or
    /// expert width the kernels do not take, stores or scratch past the
    /// plan's terms, and every refusal the sites, the routed blocks, the host
    /// tier and the expert tier's open make.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<MistralLarge4Model, GpuError> {
        Body::open_placed_with(file, plan, inputs, card, host, Residency::Off)
    }

    /// [`Body::open_placed`] under `lever`: [`Residency::Off`] keeps the load's slot map for the
    /// model's life; a [`Residency::Mid`] also holds each layer's churn pool in the host set and
    /// starts the machine over the slot map once the body's pieces are sized, which the prompt
    /// calls' host streaming ([`Body::set_xstream`]) runs on. The host set is read in
    /// (`HostCfg::populate`) for the machine's source to stage from.
    ///
    /// # Errors
    /// Those of [`Body::open_placed`], and the machine's own at its start.
    pub fn open_placed_with(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        lever: Residency,
    ) -> Result<MistralLarge4Model, GpuError> {
        let tiers = TierOpen::of_machine(plan.machine)?;
        refuse_tier_count(WHAT, tiers.len())?;
        stage_run(WHAT, plan, card, inputs.spec.layers.len())?;
        let run = hosted(&inputs.spec.layers).map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let residency = resid::spec(lever, inputs, run.len())?;
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            residency,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set, glue| {
                Body::load(
                    gpu,
                    file,
                    w,
                    (plan, inputs, card),
                    host,
                    (set, glue),
                    &tiers,
                )
            },
        )
    }

    /// The body of a placed load (`open_placed`'s arguments as the loader
    /// hands them on).
    #[allow(
        clippy::too_many_lines,
        reason = "the checked load: the shape's refusals, the stores against the plan, the routed blocks, the host tier, the rope table"
    )]
    fn load(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        (plan, inputs, card): (&Plan<'_>, &PlanInputs, usize),
        host_cfg: HostCfg,
        (set, glue): (HostResidency, ResidencyGlue),
        tiers: &[TierOpen],
    ) -> Result<Body, GpuError> {
        gpu.context().bind_to_thread()?;
        let split: &Split = file;
        let spec = &inputs.spec;
        let hp = &inputs.hp;
        let (_, ctx) = stage_run(WHAT, plan, card, spec.layers.len())?;
        // The context bound is the reader's own rule: the trained context
        // (the selected attention serves past the indexer's top-k).
        model::arch::mistrallarge4::spec::serve_ctx(ctx as u64, spec)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let sh = Shape::of(hp, ctx)?;
        tensors::globals(split)?;
        let stream = gpu.stream();
        let layers = (0..hp.n_layer)
            .map(|l| tensors::layer((split, w), hp, stream, l))
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
                            scratch: IndexerScratch::new(stream, IDX_STEP_ROWS, ctx)?,
                        })
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        // The stores are the plan's K/V term to the byte: the latent rows of
        // the layers that own a cache and the index keys of the full ones.
        let held: usize = stores.iter().map(LayerStore::kv_bytes).sum();
        let planned = inputs.kv.bytes(0..sh.n_layer, plan.ctx_max);
        if held as u64 != planned {
            return Err(GpuError::shape(
                WHAT,
                format!("the stores hold {held} bytes; the plan counts {planned}"),
            ));
        }
        let k = Kernels::load(gpu, &sh)?;
        let acts = Acts::new(stream, layers.iter().flat_map(tensors::sites))?;
        let mut s = Scratch::new(stream, &sh)?;
        let perm = Perm::new(stream, &sh)?;
        // The list's width never moves; the live rows beside it the refresh
        // writes.
        s.host_ints[words::TOP_K] = sh.top_k as u32;
        s.ints.copy_from_host(stream, &s.host_ints)?;
        // The rope table: YaRN at the LeRoPE factors, the attention factor
        // the form computes cancelling ggml's own magnitude scale (the
        // reader's `Yarn.magnitude` states the net, 1). The factors are the
        // file's own global tensor.
        let rope = RopeSpec::yarn(
            hp.rope_base,
            hp.yarn.factor,
            hp.yarn.orig_ctx as i32,
            hp.yarn.beta_fast,
            hp.yarn.beta_slow,
            sh.rope,
        )
        .with_factors(&tensors::file_f32(split, "rope_freqs.weight", sh.rope / 2)?)
        .map_err(|e| GpuError::plan(WHAT, e))?;
        let table = RopeTable::new(&rope).map_err(|e| GpuError::plan(WHAT, e))?;
        // The routed run: the layers past the dense lead, each with a host
        // share and the card's prefix of its ids the plan keeps.
        let cfg: Vec<LayerKind> = spec.layers.iter().map(LayerKind::of).collect();
        let run = hosted(&spec.layers).map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
        let map = SlotMap::of_plan_tiers(plan, card, &tier_cards, run.clone(), ffn::N_EXPERT)?;
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            ffn::N_EXPERT,
        )?);
        let mut routed = Routed::new_configured(
            gpu,
            split,
            w,
            (&spec.layers, &map, &slots),
            RoutedDims {
                embd: sh.hidden,
                rms_eps: sh.eps,
                batch_rows: T_MAX,
            },
            ffn::CARD,
            ffn::CONFIG,
        )?;
        // The step's routed buffers, the host boundary and the expert tier's
        // page hold a verify's rows in flight.
        routed.arm_rows(gpu, &map, rows::PASS_ROWS)?;
        // The plan reserves the tier's batch legs for blocks of the plan's prompt
        // rows (`PROMPT_ROWS`, which is the prompt batch's unit `T_MAX`), so the hang
        // makes them at that width: the reserve is checked against what the hang holds.
        let hang = Hang {
            tiers,
            family: &tier::NODE_FAMILY,
            open: LegOpen {
                n_expert: ffn::N_EXPERT,
                layers: cfg.len(),
                hidden: sh.hidden,
                n_used: ffn::N_USED,
                rows: rows::PASS_ROWS,
            },
            cols: PROMPT_ROWS,
            card_dontneed: host_cfg.card_dontneed,
            bytes: CardBytes::File,
        };
        let mut hybrid = hosted_hybrid_tiered(
            (gpu, file, plan),
            (run.clone(), map),
            (host_cfg, set, &glue),
            (
                HostWidths {
                    embd: sh.hidden,
                    ff: sh.ff,
                    n_used: sh.used,
                },
                rows::PASS_ROWS,
            ),
            Some(&hang),
            |src| host::routed_layers(src, hp, spec, run.clone()),
        )?;
        prompt_walk::prepare_host(gpu, &mut hybrid, T_MAX)?;
        let embed = EmbedTable::new(Arc::clone(file), &names::token_embd(), sh.hidden)?;
        if embed.n_vocab() != hp.n_vocab {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the embedding holds {} rows, the vocabulary {}",
                    embed.n_vocab(),
                    hp.n_vocab
                ),
            ));
        }
        let prompt_reserve = inputs
            .prompt_batch_bytes(plan.machine)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let prefill = Prefill::new(gpu, &sh, &layers)?;
        let pass_rows = rows::RowSet::new(stream, &sh, &stores)?;
        let mut units = BatchBufs::new(gpu, sh.hidden, T_MAX, 1)?;
        units.set_softcap(sh.softcap)?;
        let mut body = Body {
            layers,
            cfg,
            stores,
            k,
            acts,
            perm,
            s,
            x: DeviceBuffer::zeroed(stream, sh.hidden)?,
            table,
            taps: None,
            forced: None,
            feat: None,
            ffn: routed,
            shell: TierShell::new(hybrid, glue, sh.hidden, WHAT),
            slots,
            embed,
            row: vec![0.0; sh.hidden],
            seat: seat::State::of_load(gpu, file, plan, card, (planned, ctx))?,
            pass_rows,
            units,
            prefill,
            prompt_reserve,
            sh,
        };
        refuse_scores_apart(plan, card, body.index_scores_bytes())?;
        refuse_scratch_past(
            WHAT,
            plan,
            card,
            body.scratch_bytes(),
            "step rows, a verify's rows, activation forms, permutations, lists and index scores, \
             the prompt batch's index scores, routed blocks, host boundary and slot map",
        )?;
        body.check_prompt_reserve(1)?;
        body.start_residency(gpu)?;
        Ok(body)
    }

    /// The step's device buffers beside the stores and the weights, and the
    /// prompt batch's index scores (sized by the context, not the prompt
    /// reserve's rows): what the plan's scratch term and its position rows
    /// hold the load to.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.x.num_bytes()
            + self.acts.bytes()
            + self.perm.bytes()
            + self
                .stores
                .iter()
                .map(LayerStore::scratch_bytes)
                .sum::<usize>()
            + self.ffn.bytes()
            + self.shell.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
            + self.prefill.scratch_bytes()
            + self.pass_rows.bytes()
    }

    /// The indexer's scores over the context, the step's row of each full
    /// layer and the prompt batch's [`IDX_CHUNK`] rows: the part of
    /// [`Body::scratch_bytes`] that grows with the context, which the plan
    /// counts by position (`place::index_scores_row_bytes`) and the load
    /// holds equal to it.
    #[must_use]
    pub fn index_scores_bytes(&self) -> usize {
        self.stores
            .iter()
            .map(LayerStore::scores_bytes)
            .sum::<usize>()
            + self.prefill.index_scores_bytes()
    }

    /// The K/V planes the stores hold: the plan's K/V term.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(LayerStore::kv_bytes).sum()
    }

    /// The layers whose experts the card holds a prefix of (a gate's
    /// instrument).
    #[must_use]
    pub fn card_layers(&self) -> Vec<usize> {
        (0..self.cfg.len())
            .filter(|&l| self.ffn.card_has(l))
            .collect()
    }

    /// The routed experts the expert tier card holds of each layer (a gate's
    /// instrument): 0 on a layer off the tier and on every layer of a load
    /// with no tier card.
    #[must_use]
    pub fn tier_experts(&self) -> Vec<usize> {
        (0..self.cfg.len()).map(|l| self.ffn.tier_k(l)).collect()
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }
}

/// Refused by name, as the body's, when the indexer's scores the load made, `made` device bytes,
/// are not the bytes card `card` of `plan` counts by position for them
/// (`CardTotals::stage_rows_bytes`, `place::index_scores_bytes` at the plan's context): the
/// plan's flat scratch term holds none of them, so a buffer that grows with the context and is
/// not in the formula, or a formula that is not the buffers, ends the load here instead of at a
/// context the flat term happens to pass.
fn refuse_scores_apart(plan: &Plan<'_>, card: usize, made: usize) -> Result<(), GpuError> {
    let counted = plan
        .cards
        .get(card)
        .map(|c| c.stage_rows_bytes)
        .ok_or_else(|| GpuError::shape(WHAT, format!("the plan totals no card {card}")))?;
    if made as u64 != counted {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the load's indexer scores hold {made} device bytes (an f32 over the context, \
                 {IDX_STEP_ROWS} row on each full layer's step scratch and {IDX_CHUNK} on the \
                 prompt batch's); card {card}'s plan counts {counted} for the stage's position \
                 rows (`place::index_scores_row_bytes` a position)"
            ),
        ));
    }
    Ok(())
}

impl Shape {
    /// The widths of `hp`'s file at `ctx` positions, refused by name where the
    /// kernels or the router do not take them: the lightning indexer's own
    /// heads and width, an expert count other than the router's, a pick count
    /// other than the router's or past the list's width.
    fn of(hp: &Hparams, ctx: usize) -> Result<Shape, GpuError> {
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
        if (sh.experts, sh.used) != (ffn::N_EXPERT, ffn::N_USED) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the file routes {} experts and picks {}, the router serves exactly {} and {}",
                    sh.experts,
                    sh.used,
                    ffn::N_EXPERT,
                    ffn::N_USED
                ),
            ));
        }
        if sh.top_k == 0 || sh.used > sh.top_k {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the list's width {} and the router's {} picks: at most the list's width",
                    sh.top_k, sh.used
                ),
            ));
        }
        Ok(sh)
    }

    /// The rope part of a cache row: a roped layer's `rope`, none on NoPE.
    pub(crate) fn pe(&self, rope_layer: bool) -> usize {
        if rope_layer { self.rope } else { 0 }
    }
}

impl Scratch {
    /// The step's buffers, `hidden`-wide rows and the attention partials at
    /// the wider of the flash's segments and the walk's over the one-row
    /// window and the selected list (the walk reads `top_k` rows of the
    /// cache, whatever the context).
    fn new(stream: &CudaStream, sh: &Shape) -> Result<Scratch, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let width = sh.latent + sh.rope;
        let (idx_q, idx_d) = (sh.idx_heads * sh.idx_d, sh.idx_d);
        let q_rows = sh.heads;
        let segs = segments_for(sh.top_k).max(segments(1, sh.top_k));
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
            v_act: Q8Act::with_slots(stream, sh.heads, sh.latent)?,
            v_ids: DeviceBuffer::from_host(stream, &(0..sh.heads as u32).collect::<Vec<u32>>())?,
            av: z(sh.heads * sh.head_v)?,
            r: z(sh.hidden)?,
            ffn_inp: z(sh.hidden)?,
            ffn_out: z(sh.hidden)?,
            gathered: DeviceTensor::zeroed(stream, sh.top_k, width)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
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
        // kernel's tail order: the load's own row move
        // (`rows::rope_rows_to_tail`) over the head's value numbers, so value
        // `dst` of the moved head reads the file's value `src`.
        let head_move = |head: usize, rope: usize| {
            let mut rows: Vec<usize> = (0..head).collect();
            model::arch::mistrallarge4::rows::rope_rows_to_tail(&mut rows, head, rope);
            rows.into_iter().enumerate().map(|(dst, src)| (src, dst))
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
    /// The tables' device bytes.
    fn bytes(&self) -> usize {
        [&self.q, &self.k, &self.f_rope]
            .into_iter()
            .map(|(src, dst)| src.num_bytes() + dst.num_bytes())
            .sum()
    }
}

/// The attention's logit scale: `1/√key_length_mla`, the reference's own
/// (`kq_scale` of its graph — YaRN without magnitude scaling).
pub(crate) fn kq_scale(sh: Shape) -> f32 {
    1.0 / (sh.head_k as f32).sqrt()
}

/// The resident i-quant stack `name`'s word tensor.
pub(crate) fn stack<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(STEP, name)? {
        DevWeight::KQuant { w: t, .. } => Ok(t),
        _ => Err(GpuError::tensor(STEP, name, "an i-quant stack")),
    }
}

/// The down `_sel` of `family` over `down`'s stack, for the heads' up-projection:
/// the launcher the table's entry for the stack's type names (`ty` is the file's
/// type, which the i-quant launcher takes).
pub(crate) fn up_sel(
    gpu: &Gpu,
    k: &Kernels,
    (family, ty): (SelUp, GgmlType),
    down: &SelDown<'_>,
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    match family {
        SelUp::Q4k => gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            down.w,
            down.act,
            down.sel,
            down.n_slots,
            down.rows_per_expert,
            y,
        ),
        SelUp::Q5k => k.iq.enqueue_gemv_q5k_sel(stream, down, fault, y),
        SelUp::Q6k => k.q6.enqueue_gemv_q6k_sel(
            stream,
            down.w,
            down.act,
            down.sel,
            down.n_slots,
            down.rows_per_expert,
            y,
        ),
        SelUp::IQuant => k.iq.enqueue_gemv_sel(stream, ty, down, fault, y),
    }
}

/// The attention half of layer `l` (the module doc's stages 2–8): `x` in,
/// the residual's delta into `s.r`; the residual add is the caller's. A
/// `forced` list replaces the pick of a full layer's top-k before the
/// attention reads it.
#[allow(
    clippy::too_many_arguments,
    reason = "the half's launches: its card, weights and kernels, the shape, the site context, the layer and its number, the stores, the activation forms, the permutations, the buffers, the running row, a gate's forced list and the fault sink (rust-quality R8)"
)]
pub(crate) fn attention(
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
    forced: Option<&DeviceBuffer<u32>>,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let (rope, pe) = (a.kind.rope, sh.pe(a.kind.rope));
    let width = sh.latent + pe;
    // 2. the norm, and 3. the query projection: a site that reads the q8_1
    //    rows takes the norm and its q8_1 form in one launch, any other
    //    quantizes its own form from the normed row.
    let gain = w.f32_buf(STEP, &a.attn_norm)?;
    if a.q.reads() == Form::Q8x128 {
        let act = acts.q8_1_at(STEP, sh.hidden, 1)?;
        k.fused
            .enqueue_norm_quant(stream, x, gain, sh.eps, act, &mut s.xn, fault)?;
        a.q.kgemv(cx, act, l, &mut s.q)?;
    } else {
        elem.enqueue_rms_norm(stream, x, gain, sh.eps, sh.hidden, 1, &mut s.xn)?;
        a.q.project(cx, acts, l, &s.xn, &mut s.q)?;
    }
    // The rope of the query's columns.
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
            fault,
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
        // A gate's forced list stands for the pick the top-k just made.
        if let Some(list) = forced {
            st.list.copy_from_device_async(list, stream)?;
        }
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
        fault,
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
    match &a.v_b {
        VUp::Planes(name) => {
            let (qs, dd) = w.q8_planes(STEP, name)?;
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
        }
        // The heads are the experts of the stack, each reading its own column
        // of the latent outputs: slot `h` writes head `h`'s `head_v` values
        // where the per-head gemv would.
        VUp::Sel { name, ty, family } => {
            gpu.enqueue_quantize_q8_1_layer(&s.att, &mut s.v_act, l)?;
            let down = SelDown {
                w: stack(w, name)?,
                act: &s.v_act,
                sel: &s.v_ids,
                n_slots: sh.heads,
                rows_per_expert: sh.head_v,
            };
            up_sel(gpu, k, (*family, *ty), &down, fault, &mut s.av)?;
        }
    }
    a.o.project(cx, acts, l, &s.av, &mut s.r)
}

impl LayerTap {
    /// The tap's buffers, sized by the layer's kind (a rope layer's flash
    /// row is the cache's width, a NoPE layer's the latent alone; a shared
    /// layer keeps no indexer taps); a routed layer's routing copy as wide as
    /// the router's buffers and the step's ids, `route`.
    fn new(
        stream: &CudaStream,
        sh: &Shape,
        kind: Kind,
        route: Option<(&RouterOut, &DeviceBuffer<u32>)>,
    ) -> Result<LayerTap, GpuError> {
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
            route: route
                .map(|(out, ids)| -> Result<RouteTap, GpuError> {
                    Ok(RouteTap {
                        logits: z(out.logits.len())?,
                        probs: z(out.probs.len())?,
                        ids: DeviceBuffer::zeroed(stream, ids.len())?,
                        weights: z(out.weights.len())?,
                    })
                })
                .transpose()?,
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
            self.route.as_ref().map_or(0, |r| {
                r.logits.num_bytes()
                    + r.probs.num_bytes()
                    + r.ids.num_bytes()
                    + r.weights.num_bytes()
            }),
        ]
        .into_iter()
        .sum()
    }

    /// Copy the layer's compared values as the layer leaves them (the
    /// module doc's): the scaled query, the two residuals, the key path's
    /// three forms, the indexer's queries, key and weights, beside the list
    /// the layer read. Pure copies, ordered on the engine stream; the
    /// attention's own norm row is [`LayerTap::snap_attention`]'s, the
    /// block's outputs [`LayerTap::snap_dense`]'s or [`LayerTap::snap_join`]'s.
    pub(crate) fn snap(
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
    /// the ffn: the dense lead's `rms_norm` writes the same scratch.
    pub(crate) fn snap_attention(
        &mut self,
        stream: &CudaStream,
        s: &Scratch,
        sh: Shape,
    ) -> Result<(), GpuError> {
        copy_row(stream, &s.xn, &mut self.attn_norm, sh.hidden)
    }

    /// The dense lead's block output.
    pub(crate) fn snap_dense(
        &mut self,
        stream: &CudaStream,
        s: &Scratch,
        sh: Shape,
    ) -> Result<(), GpuError> {
        copy_row(stream, &s.ffn_out, &mut self.ffn_out, sh.hidden)
    }

    /// A routed layer's outputs from its join's terms, in the join's order:
    /// the routed sum `host + card` (mainline's `ffn_moe_out`), the shared
    /// output, and the block's output, their sum. Each an f32 add the join
    /// itself makes, so the copies are its bits.
    pub(crate) fn snap_join(
        &mut self,
        gpu: &Gpu,
        t: &ffn::JoinTerms<'_>,
        sh: Shape,
    ) -> Result<(), GpuError> {
        let (stream, elem, n) = (gpu.stream(), gpu.elem(), sh.hidden);
        match t.card {
            Some(card) => elem.enqueue_add(stream, t.host, card, n, &mut self.moe)?,
            None => copy_row(stream, t.host, &mut self.moe, n)?,
        }
        copy_row(stream, t.shared, &mut self.sh, n)?;
        elem.enqueue_add(stream, &self.moe, &self.sh, n, &mut self.ffn_out)
    }

    /// A routed layer's routing: the taps' router's logits, probabilities and
    /// weights, and the ids the step's own router picked.
    pub(crate) fn snap_route(
        &mut self,
        stream: &CudaStream,
        out: &RouterOut,
        ids: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let r = self.route.as_mut().ok_or(GpuError::State {
            what: STEP,
            missing: "the routing tap of a routed layer",
        })?;
        r.logits.copy_from_device_async(&out.logits, stream)?;
        r.probs.copy_from_device_async(&out.probs, stream)?;
        r.weights.copy_from_device_async(&out.weights, stream)?;
        r.ids.copy_from_device_async(ids, stream)?;
        Ok(())
    }

    /// The tap's values, the scratch copies, the list the layer read and a
    /// routed layer's routing; it synchronizes the stream.
    fn to_host(&self, stream: &CudaStream) -> Result<LayerNodes, GpuError> {
        let v = |b: &DeviceBuffer<f32>| b.to_host_vec(stream);
        let (route_logits, route_ids, route_probs, route_weights) = match &self.route {
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            Some(r) => (
                v(&r.logits)?,
                r.ids.to_host_vec(stream)?,
                v(&r.probs)?,
                v(&r.weights)?,
            ),
        };
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
            route_logits,
            route_ids,
            route_probs,
            route_weights,
        })
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Body;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> model::arch::Arch {
        model::arch::Arch::MistralLarge4
    }

    /// The token's embedding row, read from the file's table on the host.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        self.embed.row_into(token, &mut self.row)?;
        Ok(DecodeInput { token, pos })
    }

    /// The running row and the step's words, each in one copy: the
    /// embedding row, the position, the live rows, the walk's visible pair,
    /// the selected attention's count and the rope row (its f32 values and
    /// their bits for the score pass's table).
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::Body::refresh";
        let DecodeInput { pos, .. } = *input;
        let sh = self.sh;
        if pos as usize >= sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("position {pos} past the cache's {} rows", sh.ctx),
            ));
        }
        let n = (pos + 1) as usize;
        self.s.host_pos[words::POS] = pos;
        self.s.host_ints[words::N_VIS] = pos + 1;
        self.s.host_vis[0] = 0;
        self.s.host_vis[1] = n.min(sh.top_k) as u32;
        self.s.host_n_keys[words::N_KEYS] = n.min(sh.top_k) as u32;
        // The gathered flash's rows are the selection's own count
        // (`min(pos + 1, top_k)`): the flash counts the current row on top
        // of the word it reads and drops it again for the one token
        // (`m = 1`), so the word names the whole visible count.
        self.s.host_flash_n[0] = (pos + 1).min(sh.top_k as u32);
        self.s.host_cs.clear();
        self.table
            .push(pos, Direction::Forward, &mut self.s.host_cs);
        self.s.host_table_words.clear();
        self.s
            .host_table_words
            .extend(self.s.host_cs.iter().map(|f| f.to_bits()));
        self.x.copy_from_host(stream, &self.row)?;
        self.s.pos.copy_from_host(stream, &self.s.host_pos)?;
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
        // The step lays one row in the feature tap, whatever replays it.
        if let Some(f) = self.feat.as_mut() {
            f.step.plan(1);
        }
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        crate::step::enqueue(gpu, w, self, head)
    }

    /// Rewind every store to empty: each store is a plane of positions the
    /// launches read only below the position's own live count, and every
    /// step writes its position's row before it reads it, so nothing a
    /// later call reads survives a cut. The host tier settles its exchange.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.shell.hybrid.settle(gpu.stream())
    }

    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Rms
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.scratch_bytes()
            + self.prompt_bytes()
            + self.prefill_tap_bytes()
            + self.feature_bytes()
            + self
                .taps
                .as_ref()
                .map_or(0, |t| t.layers.iter().map(LayerTap::bytes).sum::<usize>())
            + self.forced.as_ref().map_or(0, |f| f.list.num_bytes())
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Body> {
        Some(self)
    }
}

impl TierBody for Body {
    type Experts = HostRun;
    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }
    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.shell.hybrid_mut()
    }
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.shell.residency_parts(&self.slots)
    }
    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.shell.hybrid.serve_captured_of(chain)
    }
}

impl Drop for Body {
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

impl bloomery_gpu::model::Instrumented for Body {
    /// Fill the first `rows` cache rows of every store with the body's own
    /// synthetic pattern — the latent rows a position-scaled ramp, the
    /// index-key rows a sign-alternating one, both inside the f16 normals
    /// and the rows past them zero: the state a prompt of `rows` tokens
    /// could have left behind, not what the model would have written. The
    /// pattern goes into the buffers the load made, never a new allocation,
    /// so a captured chain stays valid over the addresses it recorded.
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
                latent.buf_mut().copy_from_host(stream, &host)?;
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
                idx.keys.buf_mut().copy_from_host(stream, &host)?;
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
            self.v_act.device_bytes(),
            self.v_ids.num_bytes(),
            self.av.num_bytes(),
            self.r.num_bytes(),
            self.ffn_inp.num_bytes(),
            self.ffn_out.num_bytes(),
            self.gathered.buf().num_bytes(),
            self.pos.num_bytes(),
            self.ints.num_bytes(),
            self.vis.num_bytes(),
            self.n_keys.num_bytes(),
            self.flash_n.num_bytes(),
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
        if self.feat.is_some() {
            return Err(GpuError::shape(
                "mistrallarge4::arm_taps",
                "a body with a feature tap: the taps' span and injected rows would leave the \
                 feature tap's copies standing on rows the walk did not enter",
            ));
        }
        // A tier layer's card sum is made inside its join, in no buffer the
        // taps could copy: the instrument is the single-stage load's.
        if self.ffn.tier().is_some_and(TierSide::any) {
            return Err(GpuError::shape(
                "mistrallarge4::arm_taps",
                "a load with an expert tier card: a tier layer's routed sum is made inside the \
                 join, which the taps do not copy",
            ));
        }
        let stream = gpu.stream();
        let sh = self.sh;
        let router = Router::load(gpu.context())?;
        let route = Router::new_out(stream)?;
        let mut layers = Vec::with_capacity(sh.n_layer);
        for (a, kind) in self.layers.iter().zip(&self.cfg) {
            let routed = kind.host_leg().then(|| (&route, self.ffn.route_ids()));
            layers.push(LayerTap::new(stream, &sh, a.kind, routed)?);
        }
        self.taps = Some(Taps {
            layers,
            inject: Vec::new(),
            span: None,
            router,
            route,
        });
        Ok(())
    }

    /// Hold the chain to the layers of `span` (and the head only when the
    /// span ends at the model's end), `None` the whole chain. An empty or
    /// reversed span and one that ends past the model's last layer are
    /// refused by name: none of them is the whole chain. The caller holds
    /// the graph rule: a captured step replays the walk it recorded.
    pub fn set_layer_span(&mut self, span: Option<Range<usize>>) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::set_layer_span";
        if let Some(s) = &span {
            if s.start >= s.end {
                return Err(GpuError::shape(
                    WHAT,
                    format!("the span {s:?} is empty or reversed"),
                ));
            }
            if s.end > self.layers.len() {
                return Err(GpuError::shape(
                    WHAT,
                    format!("the span {s:?} is past the model's {}", self.layers.len()),
                ));
            }
        }
        match &mut self.taps {
            Some(t) => t.span = span,
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
    /// it synchronizes the stream), each routed layer's routing among them.
    /// Refused by name: a range past the model's layers.
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
        layers.map(|l| taps.layers[l].to_host(stream)).collect()
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

    /// Read `list`, the `top_k` rows another path picked, as full layer
    /// `layer`'s selection at the last position of the calls that follow (a
    /// step's position, the prompt batch's last row) in place of the pick the
    /// layer's top-k makes, until [`Body::clear_forced_selection`]: the
    /// instrument that runs a position again on the other path's list, to
    /// tell an exchange at a near tie of the index scores from a defect. The
    /// layer's scores stay its own; the shared layers after it read the
    /// forced list as the layer's. The caller holds the graph rule. A verify
    /// pass refuses by name while the list stands: it is the step's one pick's.
    ///
    /// # Errors
    /// Refused by name: the per-layer taps unarmed, a layer that is not a
    /// full indexer's, a list that is not `top_k` rows of the context.
    pub fn force_selection(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        list: &[u32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "mistrallarge4::force_selection";
        if self.taps.is_none() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the per-layer taps (arm_taps): the force is their own instrument",
            });
        }
        if !self.layers.get(layer).is_some_and(|a| a.kind.full) {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer} is not a full indexer of the model"),
            ));
        }
        let (top_k, ctx) = (self.sh.top_k, self.sh.ctx);
        if list.len() != top_k || list.iter().any(|&r| r as usize >= ctx) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a list of {} rows, each inside the context's {ctx}, where the layer picks {top_k}",
                    list.len()
                ),
            ));
        }
        self.forced = Some(Forced {
            layer,
            list: DeviceBuffer::from_host(gpu.stream(), list)?,
        });
        Ok(())
    }

    /// Drop the forced list ([`Body::force_selection`]): every layer's top-k
    /// picks its own again.
    pub fn clear_forced_selection(&mut self) {
        self.forced = None;
    }

    /// One launch of the router entry the step runs, over a planted weight
    /// whose expert 0 row is NaN and the rest finite and strictly
    /// descending, at a one-hot input, raising on layer `layer`'s sink: a
    /// NaN lane must raise [`FaultSite::Router`] on the fault word and be
    /// picked by no slot (a gate's instrument; it synchronizes the stream).
    pub fn router_lane_nan(&mut self, gpu: &Gpu, layer: usize) -> Result<(bool, bool), GpuError> {
        const WHAT: &str = "mistrallarge4::router_lane_nan";
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let (hidden, n_expert, used) = (self.sh.hidden, self.sh.experts, self.sh.used);
        if !self.cfg.get(layer).is_some_and(|k| k.host_leg()) {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer} is not a routed one"),
            ));
        }
        let mut w = vec![0.0f32; n_expert * hidden];
        w[..hidden].fill(f32::NAN);
        for (r, logit) in w.chunks_mut(hidden).enumerate().skip(1) {
            logit[0] = 20.0 - r as f32;
        }
        let mut x = vec![0.0f32; hidden];
        x[0] = 1.0;
        let w_dev = DeviceTensor::upload(stream, &w, n_expert, hidden)?;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let router = Router::load(gpu.context())?;
        let mut route = Router::new_out(stream)?;
        router.route(
            stream,
            &x_dev,
            &RouteIn {
                w: &w_dev,
                bias: None,
                scale: 1.0,
                fault: gpu.layer_sink(layer)?,
            },
            &mut route,
        )?;
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
