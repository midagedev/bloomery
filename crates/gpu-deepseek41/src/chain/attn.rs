//! The attention sub-layer of the V4.1 step: HC_PRE, the query and latent
//! paths, the compressor and the index keys where the layer owns them, the
//! attention over the window and the compressed rows, the output projection
//! and HC_POST with the next fold. See [`super`] for the piece contract.
//!
//! One [`AttnChain`] serves every layer of a card: what a layer launches
//! follows from its kind ([`LayerKind`] and the plan's stream of its ratio),
//! never from its number, and the scratch between the launches is shared,
//! since the layers run one after another on one stream. Per layer:
//!
//! 1. HC_PRE of the streams the sub-layer reads — the `pre`, `post` and
//!    `comb` that HC_POST and the ffn's fold take. The body reads the fold
//!    the previous sub-layer left, never this HC_PRE's (the lag), so HC_PRE
//!    runs on the piece's [`Branch`] beside steps 2–8, forked from the
//!    sub-layer's start and joined right before HC_POST;
//! 2. the attention norm of that fold, with its q8_1 form on a layer that
//!    owns a compressor, whose q3_K projections read it, or whose q_a and kv
//!    are K-quants ([`crate::dense`]);
//! 3. on a compressor's layer: its kv projection (and its gate projection
//!    above ratio 1), the pooled row — or, at ratio 1, the token's row — with
//!    its norm, rope and f16 cache row, then the index key of the pre-rope
//!    row. These run every step; the image's group count decides whether a
//!    row and a key are written;
//! 4. the query: q_a, its norm (with its q8_1 form when q_b is a K-quant),
//!    q_b and the tail rope;
//! 5. the latent row: kv, then its norm, tail rope, f16 slot in the ring and
//!    f16 row in the ring's shadow;
//! 6. on a layer that runs the indexer: the query's projection of q_a's norm,
//!    the weights' projection of the normed input (in q8_1, quantized here on
//!    a layer that owns no compressor), the score pass and the top-k pass,
//!    which write the stream's list;
//! 7. the attention over the window ⧺ the stream's compressed rows read
//!    through the list, then the inverse rope of its output;
//! 8. wo_a, its groups in one launch (`q8_0_gemv_heads`, or the q8_1 form of
//!    the heads and `dense`'s Q3_K block diagonal), and wo_b (with its
//!    input's q8_1 form first when it is a K-quant);
//! 9. HC_POST and the ffn's fold in one launch.
//!
//! Projections that read the same q8_1 activation run as one launch where
//! [`join_projections`] joined their Q3_K rows at load: q_a, kv and the
//! indexer's weights projection, launched before step 4 (steps 4, 5 and 6
//! then launch none of theirs); q_b with the indexer's query projection in
//! step 4; a gated compressor's kv and gate in step 3. The launch is the
//! same `q3k_gemv` over the same rows — each output row the bits its own
//! tensor's launch writes — into one buffer of which each projection's
//! output is a part ([`PartedBuffer`]). Any other format keeps the separate
//! launches.
//!
//! A layer attends the rows its stream's source layer writes
//! (`stream.kv_source`) and the list its top-k source layer selected this
//! step (`stream.topk_source`); the caller passes that layer's buffers
//! ([`Compressed`], [`Selection`]). The launches are the same at every
//! depth: while the visible count is at most `top_k`, the score pass does
//! nothing and the list is the identity `0..n_vis`, over which the
//! selected-row attention is the prefix attention bit for bit. `top_k` is a
//! load-time constant — the stride of the list and a word of the piece's own
//! copy, which the indexer reads.
//!
//! The step's integers and rope tables come from the step image, whose
//! layout is not the launches': the attention takes a token's window length
//! and compressed count as one pair, the compressor its [`CompGeom`] words.
//! [`AttnChain::enqueue_step`] gathers them once a step, one
//! `StepKernels::enqueue_gather` launch, into the piece's own words in the
//! layouts the launches read ([`WordsLayout`]); every layer then reads
//! windows of those words. The gather's pairs are fixed at load from the
//! image's layout.

use std::marker::PhantomData;
use std::mem::{ManuallyDrop, size_of};
use std::ops::{Deref, Range};

use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Branch, DeviceTensor, FaultSink, Gpu, GpuError, PartedBuffer, Q8Act, window};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::names;
use model::arch::deepseek41::plan::Planner;

use crate::attn::{self as attn_op, AttnArgs, AttnKernels, CommitArgs, SelectedRows, Staged};
use crate::compress::{
    self, CompGeom, CompressKernels, PoolArgs, RowsArgs, W_GROUPS, W_PERSISTS, w_persist, w_read,
    w_row,
};
use crate::dense::{Dense, DenseKernels, Q3kHeadsMcolArgs};
use crate::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcParams, HcPostArgs, HcPreArgs,
    HcPreScratch,
};
use crate::index_key::{self, IndexKeyArgs, IndexKeyKernels};
use crate::indexer::{self, IndexerArgs, IndexerKernels, IndexerScratch};
use crate::params::{ImageLayout, Table};
use crate::rope::{KvAppendArgs, RopeKernels, TailShape};
use crate::transpose::TransposeKernels;

const WHAT: &str = "deepseek41 AttnChain";

/// The compressed rows a layer's attention reads, and what it writes.
pub enum Compressed<'a> {
    /// A window-only layer: no stream.
    None,
    /// A layer that attends a stream another layer's compressor writes: the
    /// rows of `stream.kv_source`, as they stand after that layer ran.
    Read(&'a DeviceTensor<u16>),
    /// A layer that owns its stream's compressor.
    Source(SourceIo<'a>),
}

/// A compressor's buffers: written this step, and the rows read by the same
/// layer's attention.
pub struct SourceIo<'a> {
    /// The stream's compressed rows, `⌈ctx_max / ratio⌉` of the latent width.
    pub rows: &'a mut DeviceTensor<u16>,
    /// The index keys, on a layer that owns them: as many rows of the index
    /// key width.
    pub keys: Option<&'a mut DeviceTensor<u16>>,
    /// Above ratio 1, the compressor's ring: the `ratio` latest projections,
    /// values and scores.
    pub ring: Option<(&'a mut DeviceTensor<f32>, &'a mut DeviceTensor<f32>)>,
}

/// The buffers one layer's attention sub-layer shares with the rest of the
/// step.
pub struct AttnIo<'a> {
    /// The hyper-connection streams the sub-layer reads: `streams · n_embd`.
    pub streams_in: &'a DeviceBuffer<f32>,
    /// The folded input the previous sub-layer left: `n_embd`.
    pub fold_in: &'a DeviceBuffer<f32>,
    /// The streams after HC_POST.
    pub streams_out: &'a mut DeviceBuffer<f32>,
    /// The ffn's folded input.
    pub fold_out: &'a mut DeviceBuffer<f32>,
    /// The layer's raw window ring: `min(ctx_max, window)` latent rows in f16.
    pub ring: &'a mut DeviceTensor<u16>,
    /// The ring's shadow: `ctx_max` latent rows in f16, one per position. The
    /// append writes it; nothing in the step reads it.
    pub shadow: &'a mut DeviceTensor<u16>,
    pub compressed: Compressed<'a>,
    pub selection: Selection<'a>,
}

/// The list a layer's attention reads its compressed rows through:
/// [`AttnChain::list_len`] entries, of which the first `min(n_vis, top_k)`
/// are live.
pub enum Selection<'a> {
    /// A window-only layer.
    None,
    /// A layer that attends the list its stream's top-k source layer wrote
    /// earlier in the step.
    Read(&'a DeviceBuffer<u32>),
    /// A layer that runs the indexer: it scores its key source's index keys
    /// and writes `list`. `keys` are those keys when another layer owns them;
    /// `None` when the layer owns them, and they come in its [`SourceIo`].
    Run {
        keys: Option<&'a DeviceTensor<u16>>,
        list: &'a mut DeviceBuffer<u32>,
    },
}

/// The buffers [`AttnChain::enqueue_layer_part`] reads and writes: the
/// folded input the previous sub-layer left, the layer's ring and its
/// shadow, and its compressed rows as [`AttnIo::compressed`] names them.
pub struct PartIo<'a> {
    pub fold_in: &'a DeviceBuffer<f32>,
    pub ring: &'a mut DeviceTensor<u16>,
    pub shadow: &'a mut DeviceTensor<u16>,
    pub compressed: Compressed<'a>,
}

/// Where a chunk of a prompt batch stages its latent rows
/// ([`AttnChain::enqueue_layer_staged`]): a ring of `rows.rows()` rows — the
/// batch's chunk size — position `p` in row `p % rows.rows()`, which the
/// chunk's append writes and its attention reads beside the window ring
/// until its commit copies them into it.
pub struct StageIo<'a> {
    pub rows: &'a mut DeviceTensor<u16>,
    /// The chunk's first position, as the host planned it: the staging row
    /// of its first token is `first % rows.rows()`, and none of its tokens
    /// passes the staging's last row.
    pub first: usize,
}

/// Where the words the launches read sit in the piece's copy, in u32 words;
/// fixed at load, like the image's own layout.
#[derive(Clone, Debug)]
pub struct WordsLayout {
    /// Each token's position, in order: the latent append's; token 0's is a
    /// staged chunk's first.
    pub pos: usize,
    /// A window-only layer's attention counts, per token: the window length,
    /// then 0.
    pub window_vis: usize,
    /// The rope tables, in [`Table::ALL`] order: each the tokens' tables in
    /// token order.
    pub tables: [usize; 4],
    /// The indexer's `top_k`: written at load, never gathered.
    pub top_k: usize,
    /// Per stream of the plan, in its order.
    pub streams: Vec<StreamWords>,
    /// Words in the copy.
    pub len: usize,
}

/// One stream's words in the piece's copy.
#[derive(Clone, Debug)]
pub struct StreamWords {
    /// The geometry its compressor launches with.
    pub geom: CompGeom,
    /// Its layers' attention counts, per token: the window length, then the
    /// compressed rows visible.
    pub vis: usize,
    /// The compressed rows each token sees, in token order — the indexer's
    /// counts. One token's is the second word of its pair.
    pub nvis: usize,
    /// The compressor's step words, [`CompGeom::words`] of them.
    pub step: usize,
    /// The compressor's rope tables: a row table per group slot above ratio
    /// 1, the token's YaRN table at ratio 1.
    pub cs: usize,
}

/// The piece's intermediate buffers, as the last enqueued layer left them —
/// scratch that the next layer overwrites; what a gate reads node by node.
pub struct AttnTaps<'a> {
    /// HC_PRE's result: `pre`, `post`, then `comb` row-major.
    pub hc: &'a DeviceBuffer<f32>,
    /// The attention norm of the folded input.
    pub normed: &'a DeviceBuffer<f32>,
    /// q_a, then its norm.
    pub q_a: &'a DeviceBuffer<f32>,
    pub q_a_normed: &'a DeviceBuffer<f32>,
    /// The query heads after the tail rope.
    pub q: &'a DeviceBuffer<f32>,
    /// The latent projection, and the row normed and roped in f32.
    pub kv: &'a DeviceBuffer<f32>,
    pub kv_row: &'a DeviceBuffer<f32>,
    /// The attention output after the inverse rope.
    pub y: &'a DeviceBuffer<f32>,
    /// wo_a's output, group `g` at `g · o_lora_rank`, and wo_b's.
    pub wo_a: &'a DeviceBuffer<f32>,
    pub out: &'a DeviceBuffer<f32>,
    /// On a compressor's layer: its projections, the pre-rope row and the
    /// index key's projection.
    pub source: Option<SourceTaps<'a>>,
    /// What the last indexer layer enqueued left.
    pub select: Option<SelectTaps<'a>>,
}

/// The indexer's buffers as the last indexer layer left them.
pub struct SelectTaps<'a> {
    /// The plan stream it scored.
    pub stream: usize,
    /// The query after its rope and transform, `HEADS · HEAD_DIM`.
    pub q: &'a DeviceBuffer<f32>,
    /// The scaled weights, `HEADS`.
    pub w: &'a DeviceBuffer<f32>,
    /// The scores of the rows below `n_vis`, on a layer that selected.
    pub scores: &'a DeviceBuffer<f32>,
}

/// A compressor layer's intermediate buffers.
pub struct SourceTaps<'a> {
    pub kv: &'a DeviceBuffer<f32>,
    pub score: &'a DeviceBuffer<f32>,
    pub pre: &'a DeviceBuffer<f32>,
    pub key: &'a DeviceBuffer<f32>,
}

/// `len` values of `T` at word `off` of a buffer, read by the launches as a
/// buffer of their own. It borrows that buffer, which therefore outlives it
/// and stays in place; dropping it frees nothing.
struct View<'a, T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    _parent: PhantomData<&'a ()>,
}

impl<T> Deref for View<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buf
    }
}

impl<T> Drop for View<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `buf` is taken once, here, and never read again.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        // The window owns no memory: its raw parts are dropped, the context
        // handle with them, and nothing is freed.
        drop(buf.into_raw_parts());
    }
}

/// A [`View`] of `len` `T` from word `off` of `parent`, both one word wide.
fn view<P, T>(parent: &DeviceBuffer<P>, off: usize, len: usize) -> Result<View<'_, T>, GpuError> {
    const { assert!(size_of::<P>() == 4 && size_of::<T>() == 4) };
    let refuse = || GpuError::Shape {
        what: WHAT,
        detail: format!(
            "a view of {len} words at {off} in a buffer of {}",
            parent.len()
        ),
    };
    if len == 0 || off.checked_add(len).is_none_or(|end| end > parent.len()) {
        return Err(refuse());
    }
    let bytes = u64::try_from(off * size_of::<P>()).map_err(|_| refuse())?;
    // SAFETY: the words lie inside `parent`'s allocation (checked above) and
    // start on a word boundary, which is `T`'s; `parent` is borrowed for the
    // view's lifetime, so it outlives the view and stays in place.
    let buf = unsafe { window::<T>(parent.cu_deviceptr() + bytes, len, parent.context()) };
    Ok(View {
        buf,
        _parent: PhantomData,
    })
}

/// The gather's pairs while they are laid out: image word → the piece's word.
#[derive(Default)]
struct Pairs {
    src: Vec<u32>,
    dst: Vec<u32>,
    len: usize,
}

impl Pairs {
    /// `n` words of the copy, from a four-word boundary.
    fn alloc(&mut self, n: usize) -> usize {
        let at = self.len.next_multiple_of(4);
        self.len = at + n;
        at
    }

    fn copy(&mut self, from: u32, to: usize) -> Result<(), GpuError> {
        let to = u32::try_from(to).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("word {to} of the step words passes u32"),
        })?;
        self.src.push(from);
        self.dst.push(to);
        Ok(())
    }

    /// A field of the copy that takes the image words `from`, in order.
    fn field(&mut self, from: &[u32]) -> Result<usize, GpuError> {
        let at = self.alloc(from.len());
        for (i, &w) in from.iter().enumerate() {
            self.copy(w, at + i)?;
        }
        Ok(at)
    }
}

/// The gather's pairs and the copy's layout for images of `layout`, streams
/// of caches sized for `ctx_max` positions. An image whose every word holds
/// its own index, read through the layout's view, names each field's
/// offsets; the image layout keeps its offset arithmetic to itself.
fn plan_words(
    layout: &ImageLayout,
    ctx_max: usize,
) -> Result<(WordsLayout, Vec<u32>, Vec<u32>), GpuError> {
    let n = u32::try_from(layout.words()).map_err(|_| GpuError::Shape {
        what: WHAT,
        detail: format!("an image of {} words", layout.words()),
    })?;
    let probe: Vec<u32> = (0..n).collect();
    let img = layout.view(&probe)?;
    let dims = layout.dims();
    let (nd, m) = (dims.rope_dims, dims.tokens);
    let mut p = Pairs::default();
    let toks: Vec<_> = (0..m).map(|t| img.token(t)).collect();
    let pos = p.field(&toks.iter().map(|t| t.pos).collect::<Vec<_>>())?;
    let window_vis = p.alloc(2 * m);
    for (t, tok) in toks.iter().enumerate() {
        p.copy(tok.len, window_vis + 2 * t)?;
    }
    let mut tables = [0; 4];
    for (at, table) in tables.iter_mut().zip(Table::ALL) {
        *at = p.alloc(m * nd);
        for t in 0..m {
            for (j, &w) in img.table(t, table).iter().enumerate() {
                p.copy(w, *at + t * nd + j)?;
            }
        }
    }
    let top_k = p.alloc(1);
    let yarn = tables[table_index(Table::YarnForward)];
    let mut streams = Vec::with_capacity(layout.streams().len());
    for (s, sl) in layout.streams().iter().enumerate() {
        let f = img.stream(s);
        let (r, gm) = (sl.ratio as usize, sl.group_slots);
        let geom = CompGeom {
            ratio: r,
            max_groups: gm,
            tokens: m,
            rows: ctx_max.div_ceil(r),
        };
        let vis = p.alloc(2 * m);
        for (t, tok) in toks.iter().enumerate() {
            p.copy(tok.len, vis + 2 * t)?;
            p.copy(f.n_visible[t], vis + 2 * t + 1)?;
        }
        // One token's count is its pair's second word; more tokens keep
        // theirs in order too, as the indexer reads them.
        let nvis = if m == 1 {
            vis + 1
        } else {
            p.field(&f.n_visible[..m])?
        };
        let step = p.alloc(geom.words());
        p.copy(f.groups, step + W_GROUPS)?;
        for (g, &w) in f.state_write.iter().enumerate() {
            p.copy(w, step + w_row(g))?;
        }
        for (i, &w) in f.state_read.iter().enumerate() {
            p.copy(w, step + w_read(gm, r, 0, 0) + i)?;
        }
        // A ratio-1 stream keeps no ring: `CompGeom::pack` drops its kept
        // tokens, and so does the copy.
        if r > 1 {
            p.copy(f.persists, step + W_PERSISTS)?;
            for (j, &w) in f.persist_src.iter().enumerate() {
                p.copy(w, step + w_persist(gm, r, j))?;
            }
            for (j, &w) in f.persist_dst.iter().enumerate() {
                p.copy(w, step + w_persist(gm, r, r + j))?;
            }
        }
        let cs = if r > 1 {
            let at = p.alloc(gm * nd);
            for g in 0..gm {
                let t = img.row_table(s, g).ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!("stream {s} of ratio {r} has no row table {g}"),
                })?;
                for (j, &w) in t.iter().enumerate() {
                    p.copy(w, at + g * nd + j)?;
                }
            }
            at
        } else {
            yarn
        };
        streams.push(StreamWords {
            geom,
            vis,
            nvis,
            step,
            cs,
        });
    }
    let len = p.len.next_multiple_of(4);
    let words = WordsLayout {
        pos,
        window_vis,
        tables,
        top_k,
        streams,
        len,
    };
    Ok((words, p.src, p.dst))
}

/// The piece's copy of the step words before any gather: zero, but for the
/// word no gather writes, `top_k`.
fn constant_words(layout: &WordsLayout, top_k: usize) -> Result<Vec<f32>, GpuError> {
    let k = u32::try_from(top_k).map_err(|_| GpuError::Shape {
        what: WHAT,
        detail: format!("a top_k of {top_k} passes u32"),
    })?;
    let mut host = vec![0.0f32; layout.len];
    host[layout.top_k] = f32::from_bits(k);
    Ok(host)
}

/// `t`'s place in [`Table::ALL`]: its declaration order, which the const
/// block below holds equal to the order of `ALL`.
const fn table_index(t: Table) -> usize {
    t as usize
}

const _: () = {
    let mut i = 0;
    while i < Table::ALL.len() {
        assert!(
            table_index(Table::ALL[i]) == i,
            "Table::ALL lists the tables in declaration order"
        );
        i += 1;
    }
};

/// The piece's copies of the step words, one per row, and the gather that
/// fills them. A pass whose rows run one layer apart interleaves the rows'
/// layers, and each layer reads its own row's words.
struct Words {
    layout: WordsLayout,
    /// Per row, the words; u32 fields are read through u32 views.
    bufs: Vec<DeviceBuffer<f32>>,
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
    pairs: usize,
    /// Words of the image the pairs index.
    image_len: usize,
}

impl Words {
    /// Row `row`'s copy with the layout.
    fn of(&self, row: usize) -> Result<RowWords<'_>, GpuError> {
        let buf = self.bufs.get(row).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("row {row} of a piece of {} rows", self.bufs.len()),
        })?;
        Ok(RowWords {
            layout: &self.layout,
            buf,
        })
    }
}

/// One row's copy of the step words, as one layer's launches read it.
#[derive(Clone, Copy)]
struct RowWords<'a> {
    layout: &'a WordsLayout,
    buf: &'a DeviceBuffer<f32>,
}

impl<'a> RowWords<'a> {
    fn view<T>(&self, off: usize, len: usize) -> Result<View<'a, T>, GpuError> {
        view(self.buf, off, len)
    }
}

/// The sizes every layer launches with.
struct Dims {
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    /// wo_a's row width: one group's heads.
    group_k: usize,
    rope_dims: usize,
    /// `attention.layer_norm_rms_epsilon`.
    eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    /// The softmax scale, `1 / sqrt(head_dim)`.
    scale: f32,
}

/// The weight names of one layer's attention, built at load.
struct Names {
    hc_fn: String,
    hc_scale: String,
    hc_base: String,
    norm: String,
    q_a: String,
    q_a_norm: String,
    q_b: String,
    kv: String,
    kv_norm: String,
    sinks: String,
    out_a: String,
    out_b: String,
    /// The derived row join of q_a, kv (and on an indexer layer the
    /// indexer's weights projection) that [`join_projections`] files when
    /// they are Q3_K.
    qkv: String,
    /// The derived row join of q_b and, on an indexer layer, the indexer's
    /// query projection.
    q: String,
}

/// The compressor a layer owns.
struct SourcePlan {
    /// The plan stream it writes.
    stream: usize,
    kv: String,
    /// Its gate projection, above ratio 1.
    gate: Option<String>,
    /// The derived row join of kv and the gate, above ratio 1.
    kv_gate: String,
    norm: String,
    /// The index key's projection and norm, on a layer that owns index keys.
    keys: Option<(String, String)>,
}

/// The indexer a layer runs: its two projections, and whether it owns the
/// index keys it scores.
struct IndexerPlan {
    q_b: String,
    proj: String,
    owns_keys: bool,
}

/// One layer as the piece runs it.
struct LayerPlan {
    names: Names,
    /// The plan stream the layer attends; `None` on a window-only layer.
    stream: Option<usize>,
    source: Option<SourcePlan>,
    indexer: Option<IndexerPlan>,
    /// Word offsets of its forward and back rope tables: YaRN on a layer
    /// with a stream, the window rope otherwise.
    forward: usize,
    back: usize,
}

struct Kernels {
    hc: HcKernels,
    fused: FusedKernels,
    rope: RopeKernels,
    attn: AttnKernels,
    comp: CompressKernels,
    key: IndexKeyKernels,
    /// Loaded when a layer of the card runs the indexer.
    index: Option<IndexerKernels>,
    step: StepKernels,
    dense: DenseKernels,
    transpose: TransposeKernels,
}

/// The indexer's scratch, present when a layer of the card runs it. Its
/// projections' outputs are parts of the piece's ([`INDEX_W`], [`INDEX_Q`]).
struct SelectScratch {
    /// Per token count `m` (index `m − 1`): the normed input in q8_1, on an
    /// indexer layer that owns no compressor (one that does has it in
    /// [`SourceScratch::act`]).
    act: Vec<Q8Act>,
    /// Per token count, per plan stream whose keys a layer scores: the score
    /// and top-k passes' scratch over the stream's rows.
    streams: Vec<Vec<Option<IndexerScratch>>>,
    /// The token count and the stream of the last indexer layer enqueued.
    last: Option<(usize, usize)>,
}

/// A compressor layer's scratch.
struct SourceScratch {
    /// Per token count `m` (index `m − 1`): the normed input in q8_1, which
    /// the q3_K projections read.
    act: Vec<Q8Act>,
    /// The kv projection ([`COMP_KV`]) and the gate's ([`COMP_SCORE`]), one
    /// allocation: a joined launch writes both.
    proj: PartedBuffer<f32, 2>,
    /// The pooled row, normed, before the rope: the index key's input.
    pre: DeviceBuffer<f32>,
    act_pre: Q8Act,
    key: DeviceBuffer<f32>,
}

/// Where a piece of more than one token a pass puts its projections before
/// their token-major copies ([`crate::transpose`]), in the gemvs' row-major
/// layout: a launch over `m` tokens writes `y[r·m + c]`, so the part of a
/// joined launch whose rows start at row `r0` of the join starts at `r0 · m`
/// ([`rows_of`]) — a place that moves with `m`, which is why these are whole
/// buffers and not the scratch's fixed parts. The indexer reads its two
/// projections here, in that layout, as it reads them at one token. A piece
/// of one token has none: its layouts agree, and its projections land in
/// place.
struct Raw {
    /// The normed input's projections: q_a, kv, the indexer's weights.
    qkv: DeviceBuffer<f32>,
    /// q_a's norm's projections: the heads, the indexer's query.
    q: DeviceBuffer<f32>,
    /// On a card with a compressor layer: its kv and gate projections, and
    /// its index key's.
    proj: Option<DeviceBuffer<f32>>,
    key: Option<DeviceBuffer<f32>>,
    /// wo_b's output.
    out: DeviceBuffer<f32>,
}

struct Scratch {
    hc_pre: HcPreScratch,
    mixes: DeviceBuffer<f32>,
    hc: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    /// q_a ([`Q_A`]), the latent projection ([`KV`]) and the indexer's
    /// weights projection ([`INDEX_W`], empty on a card without an indexer
    /// layer): the projections of the normed input, one allocation, which a
    /// joined launch writes whole.
    qkv: PartedBuffer<f32, 3>,
    q_a_normed: DeviceBuffer<f32>,
    /// The query heads ([`Q`]) and the indexer's query ([`INDEX_Q`], empty
    /// on a card without an indexer layer): the projections of q_a's norm.
    q: PartedBuffer<f32, 2>,
    /// The latent row in f32, which the append writes beside the ring.
    kv_row: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    wo_a: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    /// Per token count `m` (index `m − 1`), the q8_1 forms the K-quant
    /// projections read ([`crate::dense`]): the normed input on a layer that
    /// owns no compressor, q_a's norm, the heads after the inverse rope (one
    /// column per output group and token) and wo_a.
    acts: Vec<Acts>,
    /// Present when a layer of the card owns a compressor.
    source: Option<SourceScratch>,
    select: Option<SelectScratch>,
    /// Present on a piece of more than one token.
    raw: Option<Raw>,
}

/// The parts of [`Scratch::qkv`], in row order of its join.
const Q_A: usize = 0;
const KV: usize = 1;
const INDEX_W: usize = 2;
/// The parts of [`Scratch::q`].
const Q: usize = 0;
const INDEX_Q: usize = 1;
/// The parts of [`SourceScratch::proj`].
const COMP_KV: usize = 0;
const COMP_SCORE: usize = 1;

/// [`Scratch::acts`].
struct Acts {
    normed: Q8Act,
    q_a: Q8Act,
    heads: Q8Act,
    wo_a: Q8Act,
}

/// Where the normed input's q8_1 form is, for a layer's K-quant projections.
#[derive(Clone, Copy)]
enum NormedAct {
    /// Nowhere: no projection of the layer reads it.
    None,
    /// The compressor's ([`SourceScratch::act`]).
    Source,
    /// The piece's own ([`Acts::normed`]).
    Own,
}

/// The attention piece of a card's layers: see the module comment.
pub struct AttnChain {
    layers: Range<usize>,
    plans: Vec<LayerPlan>,
    dims: Dims,
    kernels: Kernels,
    words: Words,
    scratch: Scratch,
    /// The stream HC_PRE runs on beside the sub-layer's body. One for the
    /// piece: every fork is joined before the layer's HC_POST, so HC_PRE of
    /// two layers (or rows) never overlap and share [`Scratch::hc_pre`].
    branch: Branch,
    /// The list's stride and the indexer's `top_k`, at most `list_len`.
    top_k: usize,
    /// Entries of a list per token: the file's `top_k`.
    list_len: usize,
    /// Tokens of the image the piece reads: the most a pass runs.
    tokens: usize,
}

/// Device bytes of a [`Q8Act`] of `m` columns of `k`: its allocation's
/// formula (`Q8Act::with_k`), which exposes no size of its own.
fn q8act_bytes(m: usize, k: usize) -> usize {
    let n_sb = k / 256;
    m * (64 * n_sb.div_ceil(2) * size_of::<u64>()
        + 256 * n_sb.div_ceil(4) * 4
        + 128 * n_sb.div_ceil(2) * 4
        + 8 * n_sb * 4
        + 2 * n_sb * 4)
}

/// Device bytes of an [`HcPreScratch`] for inputs of `k` values: its
/// allocation's formula, which exposes no size of its own.
fn hc_pre_bytes(k: usize) -> usize {
    let n_pieces = k / HC_PIECE;
    4 * (HC_MIX * n_pieces * HC_MAX_TOKENS + n_pieces * HC_MAX_TOKENS + 1)
}

impl AttnChain {
    /// The piece for `layers` of the model `hp` describes, reading images of
    /// `layout` and caches sized by `planner`'s `ctx_max`: every name,
    /// layer kind, table offset and gather pair resolved, the kernels loaded
    /// and the scratch allocated. One row of step words. Load-time only.
    pub fn new(
        gpu: &Gpu,
        hp: &Hparams,
        layers: Range<usize>,
        layout: &ImageLayout,
        planner: &Planner,
    ) -> Result<AttnChain, GpuError> {
        AttnChain::with_rows(gpu, hp, layers, layout, planner, 1)
    }

    /// [`AttnChain::new`] with `rows` copies of the step words, for a pass
    /// whose rows run one layer apart. The scratch is shared: a layer's
    /// launches consume it before the next layer of either row runs.
    pub fn with_rows(
        gpu: &Gpu,
        hp: &Hparams,
        layers: Range<usize>,
        layout: &ImageLayout,
        planner: &Planner,
        rows: usize,
    ) -> Result<AttnChain, GpuError> {
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        if rows == 0 {
            return Err(refuse("a piece of no rows".to_string()));
        }
        let dims = layout.dims();
        if !(1..=HC_MAX_TOKENS).contains(&dims.tokens) || dims.rope_dims != hp.rope_dims {
            return Err(refuse(format!(
                "an image of {} tokens and {} rope values; the chain runs 1..={HC_MAX_TOKENS} \
                 tokens a pass and the file's {}",
                dims.tokens, dims.rope_dims, hp.rope_dims
            )));
        }
        if hp.head_dim != attn_op::LATENT
            || hp.head_dim != compress::WIDTH
            || hp.indexer.head_dim != index_key::WIDTH
            || hp.hc.streams != HC_STREAMS
            || hp.o_groups == 0
            || !(hp.n_head * hp.head_dim).is_multiple_of(hp.o_groups)
        {
            return Err(refuse(format!(
                "latent {}, index key {}, {} streams and {} output groups over {} heads: the ops \
                 run a latent of {}, keys of {}, {HC_STREAMS} streams and whole groups",
                hp.head_dim,
                hp.indexer.head_dim,
                hp.hc.streams,
                hp.o_groups,
                hp.n_head,
                attn_op::LATENT,
                index_key::WIDTH
            )));
        }
        if layers.end > hp.n_layer || planner.stream_ratios() != dims.stream_ratios.as_slice() {
            return Err(refuse(format!(
                "layers {layers:?} of {}, streams {:?} planned and {:?} in the image",
                hp.n_layer,
                planner.stream_ratios(),
                dims.stream_ratios
            )));
        }
        let ctx_max = usize::try_from(planner.ctx_max())
            .map_err(|_| refuse(format!("a ctx_max of {} passes usize", planner.ctx_max())))?;
        let (words_layout, src, dst) = plan_words(layout, ctx_max)?;
        let plans = layers
            .clone()
            .map(|l| layer_plan(hp, planner, &words_layout, l))
            .collect::<Result<Vec<_>, _>>()?;

        let list_len = hp.indexer.top_k;
        let any_stream = plans.iter().any(|p| p.stream.is_some());
        let any_indexer = plans.iter().any(|p| p.indexer.is_some());
        if any_stream && list_len == 0 {
            return Err(refuse(
                "an indexer top_k of 0: a stream's rows are read through a list".to_string(),
            ));
        }

        let ctx = gpu.context();
        let stream = gpu.stream();
        let kernels = Kernels {
            hc: HcKernels::load(ctx)?,
            fused: FusedKernels::load(ctx)?,
            rope: RopeKernels::load(ctx)?,
            attn: AttnKernels::load(ctx)?,
            comp: CompressKernels::load(ctx)?,
            key: IndexKeyKernels::load(ctx)?,
            index: any_indexer
                .then(|| IndexerKernels::load(ctx, hp))
                .transpose()?,
            step: StepKernels::load(ctx)?,
            dense: DenseKernels::load(ctx)?,
            transpose: TransposeKernels::load(ctx)?,
        };
        let pairs = src.len();
        let constant = constant_words(&words_layout, list_len)?;
        let words = Words {
            bufs: (0..rows)
                .map(|_| DeviceBuffer::from_host(stream, &constant))
                .collect::<Result<Vec<_>, _>>()?,
            src: DeviceBuffer::from_host(stream, &src)?,
            dst: DeviceBuffer::from_host(stream, &dst)?,
            layout: words_layout,
            pairs,
            image_len: layout.words(),
        };

        let m = layout.dims().tokens;
        let q_rows = m * hp.n_head;
        let window_rows = ctx_max.min(hp.window);
        // A stream's rows are read through a list of `top_k` entries: its
        // share of the grid, whatever the stream's height.
        let comp_keys = if any_stream { list_len } else { 0 };
        let segs = attn_op::segments(window_rows, comp_keys);
        let gm = words
            .layout
            .streams
            .iter()
            .map(|s| s.geom.max_groups)
            .max()
            .unwrap_or(1);
        // Per token count 1..=m: a prompt batch's chunks run every count up
        // to the image's; a decode piece holds the one count.
        let counts = 1..=m;
        let source = if plans.iter().any(|p| p.source.is_some()) {
            Some(SourceScratch {
                act: counts
                    .clone()
                    .map(|c| Q8Act::with_k(stream, c, hp.n_embd))
                    .collect::<Result<_, _>>()?,
                proj: PartedBuffer::zeroed(stream, [m * compress::WIDTH; 2])?,
                pre: DeviceBuffer::zeroed(stream, gm * compress::WIDTH)?,
                act_pre: Q8Act::with_k(stream, gm, compress::WIDTH)?,
                key: DeviceBuffer::zeroed(stream, gm * index_key::WIDTH)?,
            })
        } else {
            None
        };
        let select = if any_indexer {
            let mut indexed = vec![false; words.layout.streams.len()];
            for p in plans.iter().filter(|p| p.indexer.is_some()) {
                let s = p
                    .stream
                    .ok_or_else(|| refuse("an indexer layer that attends no stream".to_string()))?;
                indexed[s] = true;
            }
            let streams = counts
                .clone()
                .map(|c| {
                    indexed
                        .iter()
                        .zip(&words.layout.streams)
                        .map(|(&on, sw)| {
                            on.then(|| IndexerScratch::new(stream, c, sw.geom.rows))
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(SelectScratch {
                act: counts
                    .clone()
                    .map(|c| Q8Act::with_k(stream, c, hp.n_embd))
                    .collect::<Result<_, _>>()?,
                streams,
                last: None,
            })
        } else {
            None
        };
        // The indexer's parts of the joined outputs: a token's on a card
        // with an indexer layer, none otherwise.
        let index_rows = if any_indexer { m } else { 0 };
        let qkv_parts = [
            m * hp.q_lora_rank,
            m * hp.head_dim,
            index_rows * indexer::HEADS,
        ];
        let q_parts = [
            q_rows * hp.head_dim,
            index_rows * indexer::HEADS * indexer::HEAD_DIM,
        ];
        let group_k = hp.n_head * hp.head_dim / hp.o_groups;
        let acts = counts
            .clone()
            .map(|c| {
                // A heads column per output group and token: past eight
                // columns only the per-slot allocation takes them.
                let heads = c * hp.o_groups;
                Ok(Acts {
                    normed: Q8Act::with_k(stream, c, hp.n_embd)?,
                    q_a: Q8Act::with_k(stream, c, hp.q_lora_rank)?,
                    heads: if heads <= 8 {
                        Q8Act::with_k(stream, heads, group_k)?
                    } else {
                        Q8Act::with_slots(stream, heads, group_k)?
                    },
                    wo_a: Q8Act::with_k(stream, c, hp.o_groups * hp.o_lora_rank)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let raw = if m > 1 {
            Some(Raw {
                qkv: DeviceBuffer::zeroed(stream, qkv_parts.iter().sum())?,
                q: DeviceBuffer::zeroed(stream, q_parts.iter().sum())?,
                proj: source
                    .as_ref()
                    .map(|_| DeviceBuffer::zeroed(stream, 2 * m * compress::WIDTH))
                    .transpose()?,
                key: source
                    .as_ref()
                    .map(|_| DeviceBuffer::zeroed(stream, gm * index_key::WIDTH))
                    .transpose()?,
                out: DeviceBuffer::zeroed(stream, m * hp.n_embd)?,
            })
        } else {
            None
        };
        let scratch = Scratch {
            hc_pre: HcPreScratch::new(stream, HC_STREAMS * hp.n_embd)?,
            mixes: DeviceBuffer::zeroed(stream, HC_MIX * m)?,
            hc: DeviceBuffer::zeroed(stream, HC_MIX * m)?,
            normed: DeviceBuffer::zeroed(stream, m * hp.n_embd)?,
            qkv: PartedBuffer::zeroed(stream, qkv_parts)?,
            q_a_normed: DeviceBuffer::zeroed(stream, m * hp.q_lora_rank)?,
            q: PartedBuffer::zeroed(stream, q_parts)?,
            kv_row: DeviceBuffer::zeroed(stream, m * hp.head_dim)?,
            part_v: DeviceBuffer::zeroed(stream, attn_op::partials_v_len(q_rows, segs))?,
            part_ms: DeviceBuffer::zeroed(stream, attn_op::partials_ms_len(q_rows, segs))?,
            y: DeviceBuffer::zeroed(stream, q_rows * hp.head_dim)?,
            wo_a: DeviceBuffer::zeroed(stream, m * hp.o_groups * hp.o_lora_rank)?,
            out: DeviceBuffer::zeroed(stream, m * hp.n_embd)?,
            acts,
            source,
            select,
            raw,
        };
        let dims = Dims {
            n_embd: hp.n_embd,
            n_head: hp.n_head,
            head_dim: hp.head_dim,
            q_lora_rank: hp.q_lora_rank,
            o_lora_rank: hp.o_lora_rank,
            group_k: hp.n_head * hp.head_dim / hp.o_groups,
            rope_dims: hp.rope_dims,
            eps: hp.rms_eps,
            hc_eps: hp.hc.eps,
            hc_iters: u32::try_from(hp.hc.sinkhorn_iters)
                .map_err(|_| refuse(format!("{} Sinkhorn iterations", hp.hc.sinkhorn_iters)))?,
            scale: 1.0 / (hp.head_dim as f32).sqrt(),
        };
        Ok(AttnChain {
            layers,
            plans,
            dims,
            kernels,
            words,
            scratch,
            branch: Branch::new(ctx)?,
            top_k: list_len,
            list_len,
            tokens: m,
        })
    }

    /// Entries of the list a stream's layers read, per token: the file's
    /// `top_k`. The caller allocates each list of [`AttnChain::tokens`]
    /// times this.
    #[must_use]
    pub fn list_len(&self) -> usize {
        self.list_len
    }

    /// Tokens a pass runs at most: the image's.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The indexer's `top_k` and the list's stride.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Select `top_k` rows instead of the file's, as ik's
    /// `--override-kv <arch>.attention.indexer.top_k` does: at least 1, at
    /// most [`list_len`](Self::list_len). Load-time only — one host-to-device
    /// copy of the step words (the next [`enqueue_step`](Self::enqueue_step)
    /// gathers the rest again); a step captured before it keeps the old
    /// stride and must be captured again.
    pub fn set_top_k(&mut self, gpu: &Gpu, top_k: usize) -> Result<(), GpuError> {
        if top_k == 0 || top_k > self.list_len {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a top_k of {top_k}: at least 1 and at most the lists' {} entries",
                    self.list_len
                ),
            });
        }
        let host = constant_words(&self.words.layout, top_k)?;
        for buf in &mut self.words.bufs {
            buf.copy_from_host(gpu.stream(), &host)?;
        }
        self.top_k = top_k;
        Ok(())
    }

    /// Device bytes the piece holds besides the weights and the buffers it
    /// is passed: its scratch, its copy of the step words and the gather's
    /// pairs.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let s = &self.scratch;
        let d = &self.dims;
        let plain = [
            &s.mixes,
            &s.hc,
            &s.normed,
            s.qkv.whole(),
            &s.q_a_normed,
            s.q.whole(),
            &s.kv_row,
            &s.part_v,
            &s.part_ms,
            &s.y,
            &s.wo_a,
            &s.out,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self
                .words
                .bufs
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>();
        let source = s.source.as_ref().map_or(0, |x| {
            [x.proj.whole(), &x.pre, &x.key]
                .iter()
                .map(|b| b.num_bytes())
                .sum::<usize>()
                + x.act
                    .iter()
                    .map(|a| q8act_bytes(a.m(), d.n_embd))
                    .sum::<usize>()
                + q8act_bytes(x.act_pre.m(), compress::WIDTH)
        });
        let select = s.select.as_ref().map_or(0, |x| {
            x.act
                .iter()
                .map(|a| q8act_bytes(a.m(), d.n_embd))
                .sum::<usize>()
                + x.streams
                    .iter()
                    .flatten()
                    .flatten()
                    .map(IndexerScratch::device_bytes)
                    .sum::<usize>()
        });
        let acts = s
            .acts
            .iter()
            .map(|a| {
                q8act_bytes(a.normed.m(), d.n_embd)
                    + q8act_bytes(a.q_a.m(), d.q_lora_rank)
                    + q8act_bytes(a.heads.m(), d.group_k)
                    + q8act_bytes(a.wo_a.m(), a.wo_a.n_sb() * 256)
            })
            .sum::<usize>();
        let raw = s.raw.as_ref().map_or(0, |r| {
            r.qkv.num_bytes()
                + r.q.num_bytes()
                + r.proj.as_ref().map_or(0, DeviceBuffer::num_bytes)
                + r.key.as_ref().map_or(0, DeviceBuffer::num_bytes)
                + r.out.num_bytes()
        });
        plain
            + source
            + select
            + acts
            + raw
            + self.words.src.num_bytes()
            + self.words.dst.num_bytes()
            + hc_pre_bytes(s.hc_pre.k())
    }

    /// Where the step words sit in the piece's copy.
    #[must_use]
    pub fn words_layout(&self) -> &WordsLayout {
        &self.words.layout
    }

    /// Row 0's copy of the step words, as the last
    /// [`enqueue_step`](Self::enqueue_step) left it: u32 values in f32
    /// cells, at [`words_layout`](Self::words_layout)'s offsets.
    #[must_use]
    pub fn words(&self) -> &DeviceBuffer<f32> {
        &self.words.bufs[0]
    }

    /// Rows of step words the piece holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.words.bufs.len()
    }

    /// Device bytes of one row's copy of the step words.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.words.bufs[0].num_bytes()
    }

    /// The intermediate buffers of the last enqueued layer.
    #[must_use]
    pub fn taps(&self) -> AttnTaps<'_> {
        let s = &self.scratch;
        AttnTaps {
            hc: &s.hc,
            normed: &s.normed,
            q_a: s.qkv.part(Q_A),
            q_a_normed: &s.q_a_normed,
            q: s.q.part(Q),
            kv: s.qkv.part(KV),
            kv_row: &s.kv_row,
            y: &s.y,
            wo_a: &s.wo_a,
            out: &s.out,
            source: s.source.as_ref().map(|x| SourceTaps {
                kv: x.proj.part(COMP_KV),
                score: x.proj.part(COMP_SCORE),
                pre: &x.pre,
                key: &x.key,
            }),
            select: s.select.as_ref().and_then(|x| {
                let (m, stream) = x.last?;
                let scratch = x.streams.get(m - 1)?.get(stream)?.as_ref()?;
                Some(SelectTaps {
                    stream,
                    q: &scratch.q,
                    w: &scratch.w,
                    scores: &scratch.scores,
                })
            }),
        }
    }

    /// Enqueue the step's gather: the words the layers read, from `image`,
    /// the step image's device copy, into the piece's own. Once a step,
    /// before the layers. One launch; asynchronous, allocation-free,
    /// capturable. Row 0's copy.
    pub fn enqueue_step(&mut self, gpu: &Gpu, image: &DeviceBuffer<u32>) -> Result<(), GpuError> {
        self.enqueue_step_of(gpu, image, 0)
    }

    /// [`AttnChain::enqueue_step`] into row `row`'s copy, from that row's
    /// image.
    pub fn enqueue_step_of(
        &mut self,
        gpu: &Gpu,
        image: &DeviceBuffer<u32>,
        row: usize,
    ) -> Result<(), GpuError> {
        let w = &mut self.words;
        let rows = w.bufs.len();
        let buf = w.bufs.get_mut(row).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("row {row} of a piece of {rows} rows"),
        })?;
        if image.len() != w.image_len {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "an image of {} words; the gather was laid out for {}",
                    image.len(),
                    w.image_len
                ),
            });
        }
        let x = view::<u32, f32>(image, 0, image.len())?;
        self.kernels
            .step
            .enqueue_gather(gpu.stream(), &x, &w.src, &w.dst, w.pairs, buf)
    }

    /// Enqueue layer `layer`'s attention sub-layer on `gpu`'s stream, its
    /// weights resident in `w`: the module comment's launches, reading the
    /// words the step's [`enqueue_step`](Self::enqueue_step) gathered.
    /// Asynchronous, allocation-free, capturable. Row 0's words.
    pub fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        io: AttnIo<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_layer_of(gpu, w, layer, 0, io)
    }

    /// [`AttnChain::enqueue_layer`] reading row `row`'s words, which its
    /// [`enqueue_step_of`](Self::enqueue_step_of) gathered.
    pub fn enqueue_layer_of(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        row: usize,
        io: AttnIo<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_layer_at(gpu, w, layer, row, 1, io, None)
    }

    /// Layer `layer`'s attention sub-layer for a chunk of a prompt batch:
    /// the first `m` tokens (`1 ..=` [`AttnChain::tokens`]) of row `row`'s
    /// words, token `t` at position `stage.first + t`. The chunk's latent
    /// rows go to `stage` (and their shadow rows to `io.shadow`) instead of
    /// the ring, its tokens attend the ring ⧺ those rows
    /// ([`AttnKernels::enqueue_staged`]), and one launch then commits them
    /// to `io.ring` ([`AttnKernels::enqueue_commit`]): the ring is left as
    /// `m` decode steps leave it, every output bit for bit theirs — each
    /// launch here writes, per token, what its one-token launch writes, and
    /// a projection's row-major output reaches its readers token-major
    /// ([`crate::transpose`]). Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer enqueue's arguments, the chunk's token count and its staging (rust-quality R8)"
    )]
    pub fn enqueue_layer_staged(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        row: usize,
        m: usize,
        io: AttnIo<'_>,
        stage: StageIo<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_layer_at(gpu, w, layer, row, m, io, Some(stage))
    }

    /// Layer `layer`'s latent rows alone, for a chunk of a prompt batch whose
    /// positions only later positions' windows read — the first `m` tokens
    /// (`1 ..=` [`AttnChain::tokens`]) of row `row`'s words: the attention
    /// norm of `io.fold_in`, on a compressor's layer the compressor and its
    /// index key, the latent projection and each token's row into its slot
    /// of `io.ring` and its row of `io.shadow`. Nothing else runs — no query,
    /// indexer, attention, output or HC_POST — so the chunk leaves no streams
    /// and no fold. Each launch is one [`AttnChain::enqueue_layer_staged`]
    /// makes over the same tokens (the append into the ring instead of the
    /// staging, the rows its commit would copy there), so the ring, the
    /// shadow, the compressed rows and keys and the compressor's state are
    /// its bits. Asynchronous, allocation-free.
    pub fn enqueue_layer_part(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        row: usize,
        m: usize,
        io: PartIo<'_>,
    ) -> Result<(), GpuError> {
        let lp = layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.plans.get(i))
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer} is not one of {:?}", self.layers),
            })?;
        if !(1..=self.tokens).contains(&m) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a pass of {m} tokens; the piece runs 1..={}", self.tokens),
            });
        }
        let PartIo {
            fold_in,
            ring,
            shadow,
            mut compressed,
        } = io;
        let cx = Cx {
            gpu,
            w,
            k: &self.kernels,
            d: &self.dims,
            words: self.words.of(row)?,
            layer,
            fault: gpu.layer_sink(layer)?,
            m,
        };
        let s = &mut self.scratch;
        let (d, k, stream) = (cx.d, cx.k, gpu.stream());
        let qkv = Qkv::of(w, lp, d)?;
        let normed_act = enqueue_norm(&cx, lp, s, fold_in, &mut compressed, qkv.reads_q8_1())?;
        let act = normed_act_of(&s.source, &s.acts, normed_act, m)?;
        match qkv {
            Qkv::Joint(t) => {
                let act = act.ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {layer}: a joined q_a·kv without the normed q8_1"),
                })?;
                match s.raw.as_mut() {
                    None => gpu.enqueue_gemv_q3k(t, act, s.qkv.whole_mut())?,
                    Some(raw) => {
                        gpu.enqueue_gemv_q3k(t, act, &mut raw.qkv)?;
                        let src = rows_of(&raw.qkv, d.q_lora_rank, d.head_dim, m)?;
                        k.transpose
                            .enqueue(stream, &src, d.head_dim, m, s.qkv.part_mut(KV))?;
                    }
                }
            }
            Qkv::Parts { kv, .. } => project(
                &cx,
                kv,
                &s.normed,
                act,
                d.head_dim,
                s.raw.as_mut().map(|r| &mut r.qkv),
                s.qkv.part_mut(KV),
            )?,
        }
        let pos = cx.words.view::<u32>(cx.words.layout.pos, m)?;
        let table = cx.words.view::<f32>(lp.forward, m * d.rope_dims)?;
        k.rope.enqueue_kv_norm_rope_append(
            stream,
            KvAppendArgs {
                kv: s.qkv.part(KV),
                gain: vector(w, &lp.names.kv_norm)?,
                cs: &table,
                pos: &pos,
                eps: d.eps,
                n_dims: d.rope_dims,
                m,
                out: &mut s.kv_row,
                cache: ring,
                shadow,
            },
        )
    }

    /// The layer's launches over `m` tokens: into the ring (`stage` `None`,
    /// the decode step's, one token) or through a chunk's staging.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer enqueue's arguments, the token count and the staging (rust-quality R8)"
    )]
    fn enqueue_layer_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        row: usize,
        m: usize,
        io: AttnIo<'_>,
        stage: Option<StageIo<'_>>,
    ) -> Result<(), GpuError> {
        let lp = layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.plans.get(i))
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer} is not one of {:?}", self.layers),
            })?;
        if !(1..=self.tokens).contains(&m) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a pass of {m} tokens; the piece runs 1..={}", self.tokens),
            });
        }
        let AttnIo {
            streams_in,
            fold_in,
            streams_out,
            fold_out,
            ring,
            shadow,
            mut compressed,
            selection,
        } = io;
        let top_k = self.top_k;
        let cx = Cx {
            gpu,
            w,
            k: &self.kernels,
            d: &self.dims,
            words: self.words.of(row)?,
            layer,
            fault: gpu.layer_sink(layer)?,
            m,
        };
        let s = &mut self.scratch;
        let (d, k, stream, n) = (cx.d, cx.k, gpu.stream(), &lp.names);
        let branch = &self.branch;

        let params = HcParams {
            w: q3_k(w, &n.hc_fn)?,
            scale: vector(w, &n.hc_scale)?,
            base: vector(w, &n.hc_base)?,
            eps: d.hc_eps,
            iters: d.hc_iters,
        };
        let pre = HcPreArgs {
            params: &params,
            x: streams_in,
            tokens: m,
            rms_eps: d.eps,
            fault: cx.fault,
        };
        // Only HC_POST reads what HC_PRE writes: it runs beside steps 2–8.
        let fork = branch.fork(stream)?;
        k.hc.enqueue_pre(fork.stream(), &pre, &mut s.hc_pre, &mut s.mixes, &mut s.hc)?;

        let qkv = Qkv::of(w, lp, d)?;
        let normed_act = enqueue_norm(&cx, lp, s, fold_in, &mut compressed, qkv.reads_q8_1())?;

        let q_a = match qkv {
            Qkv::Joint(t) => {
                let act = normed_act_of(&s.source, &s.acts, normed_act, m)?.ok_or_else(|| {
                    GpuError::Shape {
                        what: WHAT,
                        detail: format!("layer {layer}: a joined q_a·kv without the normed q8_1"),
                    }
                })?;
                match s.raw.as_mut() {
                    None => gpu.enqueue_gemv_q3k(t, act, s.qkv.whole_mut())?,
                    Some(raw) => {
                        gpu.enqueue_gemv_q3k(t, act, &mut raw.qkv)?;
                        for (part, r0, rows) in
                            [(Q_A, 0, d.q_lora_rank), (KV, d.q_lora_rank, d.head_dim)]
                        {
                            let src = rows_of(&raw.qkv, r0, rows, m)?;
                            k.transpose
                                .enqueue(stream, &src, rows, m, s.qkv.part_mut(part))?;
                        }
                    }
                }
                None
            }
            Qkv::Parts { q_a, .. } => Some(q_a),
        };
        enqueue_query(&cx, lp, s, q_a, normed_act)?;
        let pos = cx.words.view::<u32>(cx.words.layout.pos, m)?;
        let table = cx.words.view::<f32>(lp.forward, m * d.rope_dims)?;
        if let Qkv::Parts { kv, .. } = qkv {
            let act = normed_act_of(&s.source, &s.acts, normed_act, m)?;
            project(
                &cx,
                kv,
                &s.normed,
                act,
                d.head_dim,
                s.raw.as_mut().map(|r| &mut r.qkv),
                s.qkv.part_mut(KV),
            )?;
        }
        let mut stage = stage;
        {
            let cache: &mut DeviceTensor<u16> = match stage.as_mut() {
                Some(st) => &mut *st.rows,
                None => &mut *ring,
            };
            k.rope.enqueue_kv_norm_rope_append(
                stream,
                KvAppendArgs {
                    kv: s.qkv.part(KV),
                    gain: vector(w, &n.kv_norm)?,
                    cs: &table,
                    pos: &pos,
                    eps: d.eps,
                    n_dims: d.rope_dims,
                    m,
                    out: &mut s.kv_row,
                    cache,
                    shadow,
                },
            )?;
        }

        let (rows, vis_at) = match (&compressed, lp.stream) {
            (Compressed::Read(rows), Some(st)) => (Some(*rows), cx.words.layout.streams[st].vis),
            (Compressed::Source(io), Some(st)) => {
                (Some(&*io.rows), cx.words.layout.streams[st].vis)
            }
            _ => (None, cx.words.layout.window_vis),
        };
        let list: Option<&DeviceBuffer<u32>> = match (&lp.indexer, selection, lp.stream) {
            (None, Selection::None, None) => None,
            (None, Selection::Read(list), Some(_)) => Some(list),
            (Some(ip), Selection::Run { keys, list }, Some(st)) => {
                let keys = match (ip.owns_keys, keys, &compressed) {
                    (true, None, Compressed::Source(io)) => io.keys.as_deref(),
                    (false, Some(keys), _) => Some(keys),
                    _ => None,
                }
                .ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "layer {layer} runs the indexer and owns its keys: {}; the caller passed \
                         them elsewhere",
                        ip.owns_keys
                    ),
                })?;
                enqueue_indexer(
                    &cx,
                    lp,
                    ip,
                    &mut *s,
                    normed_act,
                    Joined {
                        weights: matches!(qkv, Qkv::Joint(_)),
                        query: joint_q3k(w, &n.q)?.is_some(),
                    },
                    IndexerIo {
                        stream: st,
                        keys,
                        list: &mut *list,
                        top_k,
                    },
                )?;
                Some(&*list)
            }
            (ip, sel, st) => {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "layer {layer} attends stream {st:?} and runs the indexer: {}; the caller \
                         passed {}",
                        ip.is_some(),
                        match sel {
                            Selection::None => "no list",
                            Selection::Read(_) => "a list to read",
                            Selection::Run { .. } => "a list to write",
                        }
                    ),
                });
            }
        };
        let vis = cx.words.view::<u32>(vis_at, 2 * m)?;
        let args = AttnArgs {
            q: s.q.part(Q),
            window: ring,
            compressed: rows,
            // A stream's rows are read through the list: while the
            // visible count is at most `top_k`, the identity.
            selected: list.map(|rows| SelectedRows {
                rows,
                stride: top_k,
            }),
            vis: &vis,
            sinks: vector(w, &n.sinks)?,
            scale: d.scale,
            tokens: m,
            heads: d.n_head,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            y: &mut s.y,
            fault: cx.fault,
        };
        match stage.as_ref() {
            None => k.attn.enqueue(stream, args)?,
            Some(st) => {
                let base = cx.words.view::<u32>(cx.words.layout.pos, 1)?;
                let staged = staging_view(st, m)?;
                k.attn.enqueue_staged(
                    stream,
                    args,
                    Staged {
                        rows: &staged,
                        base: &base,
                    },
                )?;
                let committed = k.attn.enqueue_commit(
                    stream,
                    CommitArgs {
                        staged: Staged {
                            rows: &staged,
                            base: &base,
                        },
                        tokens: m,
                        ring: &mut *ring,
                    },
                );
                DeviceTensor::release(staged);
                committed?;
            }
        }
        enqueue_output(&cx, lp, s)?;
        fork.join()?;

        let post = HcPostArgs {
            x: &s.out,
            res: streams_in,
            hc: &s.hc,
            n_embd: d.n_embd,
            tokens: m,
        };
        k.hc.enqueue_post(stream, &post, streams_out, fold_out)
    }
}

/// The attention norm of `fold_in` over the pass's tokens, with the q8_1
/// form the layer's projections read when `quantized`: on a compressor's
/// layer into the compressor's scratch, then the compressor and its index
/// key; else into the piece's own. Returns where the q8_1 form sits.
fn enqueue_norm(
    cx: &Cx<'_>,
    lp: &LayerPlan,
    s: &mut Scratch,
    fold_in: &DeviceBuffer<f32>,
    compressed: &mut Compressed<'_>,
    quantized: bool,
) -> Result<NormedAct, GpuError> {
    let (d, k, stream, w, gpu, layer) = (cx.d, cx.k, cx.gpu.stream(), cx.w, cx.gpu, cx.layer);
    let m = cx.m;
    let passed = match &*compressed {
        Compressed::None => "no compressed rows",
        Compressed::Read(_) => "rows to read",
        Compressed::Source(_) => "a compressor's buffers",
    };
    let gain = vector(w, &lp.names.norm)?;
    Ok(match (&lp.source, compressed, s.source.as_mut()) {
        (Some(sp), Compressed::Source(io), Some(src)) => {
            let act = count_of(&mut src.act, m)?;
            k.fused.enqueue_norm_quant(
                stream,
                fold_in,
                gain,
                d.eps,
                act,
                &mut s.normed,
                cx.fault,
            )?;
            enqueue_source(cx, sp, src, &mut s.raw, io)?;
            NormedAct::Source
        }
        (None, Compressed::Read(_), _) if lp.stream.is_some() && quantized => {
            k.fused.enqueue_norm_quant(
                stream,
                fold_in,
                gain,
                d.eps,
                &mut count_of(&mut s.acts, m)?.normed,
                &mut s.normed,
                cx.fault,
            )?;
            NormedAct::Own
        }
        (None, Compressed::None, _) if lp.stream.is_none() && quantized => {
            k.fused.enqueue_norm_quant(
                stream,
                fold_in,
                gain,
                d.eps,
                &mut count_of(&mut s.acts, m)?.normed,
                &mut s.normed,
                cx.fault,
            )?;
            NormedAct::Own
        }
        (None, Compressed::Read(_), _) if lp.stream.is_some() => {
            gpu.elem().enqueue_rms_norm(
                stream,
                fold_in,
                gain,
                d.eps,
                d.n_embd,
                m,
                &mut s.normed,
            )?;
            NormedAct::None
        }
        (None, Compressed::None, _) if lp.stream.is_none() => {
            gpu.elem().enqueue_rms_norm(
                stream,
                fold_in,
                gain,
                d.eps,
                d.n_embd,
                m,
                &mut s.normed,
            )?;
            NormedAct::None
        }
        _ => {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer} attends stream {:?} and owns a compressor: {}; the caller \
                     passed {passed}",
                    lp.stream,
                    lp.source.is_some(),
                ),
            });
        }
    })
}

/// The rows of `st`'s staging a chunk of `m` tokens from `st.first` fills:
/// from row `first % rows` on, which is where the append's slot `p % rows`
/// puts the chunk's first token. A window over the staging's allocation,
/// given back with [`DeviceTensor::release`]; refused when the chunk would
/// pass the staging's last row.
fn staging_view(st: &StageIo<'_>, m: usize) -> Result<ManuallyDrop<DeviceTensor<u16>>, GpuError> {
    let (rows, width) = (st.rows.rows(), st.rows.cols());
    let at = st.first % rows.max(1);
    if rows == 0 || at + m > rows {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a chunk of {m} tokens from position {} in a staging of {rows} rows: it must not \
                 pass the staging's last row",
                st.first
            ),
        });
    }
    let off = u64::try_from(at * width * size_of::<u16>()).map_err(|_| GpuError::Shape {
        what: WHAT,
        detail: format!("staging row {at} passes u64 bytes"),
    })?;
    // SAFETY: rows at .. rows of the staging's allocation (at + m <= rows,
    // checked above) are borrowed through `st` for the view's use, which ends
    // before the caller's borrow of `st` does; the view frees nothing and is
    // given back by the caller.
    Ok(unsafe {
        DeviceTensor::window(
            st.rows.buf().cu_deviceptr() + off,
            rows - at,
            width,
            st.rows.buf().context(),
        )
    })
}

/// What every launch of one layer reads besides its scratch.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    d: &'a Dims,
    words: RowWords<'a>,
    /// The model layer the launches belong to, and its fault sink.
    layer: usize,
    fault: FaultSink,
    /// Tokens of the pass.
    m: usize,
}

/// The scratch of token count `m` in `by_count` (index `m − 1`).
fn count_of<T>(by_count: &mut [T], m: usize) -> Result<&mut T, GpuError> {
    let n = by_count.len();
    m.checked_sub(1)
        .and_then(|i| by_count.get_mut(i))
        .ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("a pass of {m} tokens; the scratch holds 1..={n}"),
        })
}

/// The part of a joined launch's row-major output `whole` over `m` tokens
/// whose rows start at row `r0` of the join: `rows · m` values from `r0 ·
/// m`.
fn rows_of(
    whole: &DeviceBuffer<f32>,
    r0: usize,
    rows: usize,
    m: usize,
) -> Result<View<'_, f32>, GpuError> {
    view(whole, r0 * m, rows * m)
}

/// [`rows_of`], to write.
fn rows_of_mut(
    whole: &mut DeviceBuffer<f32>,
    r0: usize,
    rows: usize,
    m: usize,
) -> Result<crate::span::SpanMut<'_, f32>, GpuError> {
    crate::span::span_mut(WHAT, whole, r0 * m, rows * m)
}

/// The normed input's q8_1 form for `m` tokens where `at` says it is: the
/// compressor's scratch `source` or the piece's `acts`.
fn normed_act_of<'a>(
    source: &'a Option<SourceScratch>,
    acts: &'a [Acts],
    at: NormedAct,
    m: usize,
) -> Result<Option<&'a Q8Act>, GpuError> {
    let of = |v: usize| {
        m.checked_sub(1)
            .filter(|&i| i < v)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("a pass of {m} tokens; the scratch holds 1..={v}"),
            })
    };
    Ok(match at {
        NormedAct::None => None,
        NormedAct::Source => match source.as_ref() {
            Some(x) => Some(&x.act[of(x.act.len())?]),
            None => None,
        },
        NormedAct::Own => Some(&acts[of(acts.len())?].normed),
    })
}

/// A plain projection of the pass's `m` tokens (`rows` outputs each) into
/// `dst`, token-major: straight into `dst` without `raw` (a piece of one
/// token, where the layouts agree); else into the start of `raw`, the
/// gemvs' row-major layout, then copied token-major into `dst`.
fn project(
    cx: &Cx<'_>,
    d: Dense<'_>,
    x: &DeviceBuffer<f32>,
    act: Option<&Q8Act>,
    rows: usize,
    raw: Option<&mut DeviceBuffer<f32>>,
    dst: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    match raw {
        None => cx.k.dense.enqueue_m(cx.gpu, d, x, act, cx.m, dst),
        Some(raw) => {
            cx.k.dense.enqueue_m(cx.gpu, d, x, act, cx.m, raw)?;
            cx.k.transpose
                .enqueue(cx.gpu.stream(), raw, rows, cx.m, dst)
        }
    }
}

/// The query path: q_a (unless the joined launch of the normed input's
/// projections wrote it — `q_a` is `None` then), its norm, q_b (joined with
/// the indexer's query projection when [`join_projections`] filed that) and
/// the tail rope of every head. q_a's norm leaves its q8_1 form too when
/// q_b, or the layer's indexer query, reads one. At more than one token each
/// projection's row-major output is copied token-major before its reader,
/// the indexer's query excepted: the indexer reads the gemv's layout.
fn enqueue_query(
    cx: &Cx<'_>,
    lp: &LayerPlan,
    s: &mut Scratch,
    q_a: Option<Dense<'_>>,
    normed_act: NormedAct,
) -> Result<(), GpuError> {
    let (d, w, n, stream, m) = (cx.d, cx.w, &lp.names, cx.gpu.stream(), cx.m);
    let query = Query::of(w, lp, d)?;
    if let Some(q_a) = q_a {
        let act = normed_act_of(&s.source, &s.acts, normed_act, m)?;
        project(
            cx,
            q_a,
            &s.normed,
            act,
            d.q_lora_rank,
            s.raw.as_mut().map(|r| &mut r.qkv),
            s.qkv.part_mut(Q_A),
        )?;
    }
    let gain = vector(w, &n.q_a_norm)?;
    let acts = count_of(&mut s.acts, m)?;
    if query.reads_q8_1() {
        cx.k.fused.enqueue_norm_quant(
            stream,
            s.qkv.part(Q_A),
            gain,
            d.eps,
            &mut acts.q_a,
            &mut s.q_a_normed,
            cx.fault,
        )?;
    } else {
        cx.gpu.elem().enqueue_rms_norm(
            stream,
            s.qkv.part(Q_A),
            gain,
            d.eps,
            d.q_lora_rank,
            m,
            &mut s.q_a_normed,
        )?;
    }
    let q_rows = d.n_head * d.head_dim;
    match (query, s.raw.as_mut()) {
        (Query::Joint(t), None) => cx.gpu.enqueue_gemv_q3k(t, &acts.q_a, s.q.whole_mut())?,
        (Query::Joint(t), Some(raw)) => {
            cx.gpu.enqueue_gemv_q3k(t, &acts.q_a, &mut raw.q)?;
            let src = rows_of(&raw.q, 0, q_rows, m)?;
            cx.k.transpose
                .enqueue(stream, &src, q_rows, m, s.q.part_mut(Q))?;
        }
        (Query::Parts { q_b, .. }, raw) => {
            let act = q_b.reads_q8_1().then_some(&acts.q_a);
            project(
                cx,
                q_b,
                &s.q_a_normed,
                act,
                q_rows,
                raw.map(|r| &mut r.q),
                s.q.part_mut(Q),
            )?;
        }
    }
    let table = cx.words.view::<f32>(lp.forward, m * d.rope_dims)?;
    cx.k.rope
        .enqueue_rope_tail(stream, s.q.part_mut(Q), &table, heads(d, m))
}

/// Every query head of the pass's `m` tokens, its tail turned.
fn heads(d: &Dims, m: usize) -> TailShape {
    TailShape {
        width: d.head_dim,
        n_dims: d.rope_dims,
        n_vec: d.n_head,
        m,
    }
}

/// The attention output's inverse rope, wo_a and wo_b.
fn enqueue_output(cx: &Cx<'_>, lp: &LayerPlan, s: &mut Scratch) -> Result<(), GpuError> {
    let (d, w, n, stream, m) = (cx.d, cx.w, &lp.names, cx.gpu.stream(), cx.m);
    let table = cx.words.view::<f32>(lp.back, m * d.rope_dims)?;
    cx.k.rope
        .enqueue_rope_tail(stream, &mut s.y, &table, heads(d, m))?;
    let groups = d.n_head * d.head_dim / d.group_k;
    let acts = count_of(&mut s.acts, m)?;
    match Dense::of(w, &n.out_a, d.group_k, groups * d.o_lora_rank)? {
        Dense::Q8_0 { qs, d: qd } if m == 1 => cx.k.step.enqueue_q8_0_gemv_heads(
            stream,
            Q8_0GemvHeadsArgs {
                qs,
                d: qd,
                x: &s.y,
                rows_per_head: d.o_lora_rank,
                x_head_stride: d.group_k,
                y_head_stride: d.o_lora_rank,
                y_off: 0,
                y: &mut s.wo_a,
            },
        )?,
        Dense::Q3K(wa) => {
            // The groups' windows of the heads are the columns of one q8_1
            // activation of `groups` columns a token, token after token.
            cx.gpu
                .enqueue_quantize_q8_1_layer(&s.y, &mut acts.heads, cx.layer)?;
            if m == 1 {
                cx.k.dense.enqueue_q3k_heads(
                    stream,
                    wa,
                    &acts.heads,
                    d.o_lora_rank,
                    &mut s.wo_a,
                )?;
            } else {
                cx.k.dense.enqueue_q3k_heads_mcol(
                    stream,
                    Q3kHeadsMcolArgs {
                        w: wa,
                        q3: acts.heads.q3(),
                        d8: acts.heads.d8(),
                        n_sb: acts.heads.n_sb(),
                        groups,
                        rows_per_head: d.o_lora_rank,
                        m,
                        y: &mut s.wo_a,
                    },
                )?;
            }
        }
        _ => {
            return Err(GpuError::Tensor {
                what: WHAT,
                name: n.out_a.clone(),
                need: "Q3_K, or Q8_0 on a one-token pass",
            });
        }
    }
    let out_b = Dense::of(w, &n.out_b, groups * d.o_lora_rank, d.n_embd)?;
    let act = if out_b.reads_q8_1() {
        cx.gpu
            .enqueue_quantize_q8_1_layer(&s.wo_a, &mut acts.wo_a, cx.layer)?;
        Some(&acts.wo_a)
    } else {
        None
    };
    project(
        cx,
        out_b,
        &s.wo_a,
        act,
        d.n_embd,
        s.raw.as_mut().map(|r| &mut r.out),
        &mut s.out,
    )
}

/// What an indexer layer's selection reads and writes besides the scratch.
struct IndexerIo<'a> {
    /// The plan stream whose visible rows it scores.
    stream: usize,
    keys: &'a DeviceTensor<u16>,
    list: &'a mut DeviceBuffer<u32>,
    top_k: usize,
}

/// Which of an indexer layer's projections a joined launch already wrote.
#[derive(Clone, Copy)]
struct Joined {
    /// The weights' projection, with q_a and kv.
    weights: bool,
    /// The query's projection, with q_b.
    query: bool,
}

/// The indexer of an indexer layer, after the query path and the norm: the
/// query's projection of q_a's norm, the weights' projection of the normed
/// input in q8_1 (the compressor's on its layer, quantized here elsewhere)
/// — each unless `joined` says a joined launch wrote it — then the score and
/// top-k passes into the list. The visible counts and `top_k` are words of
/// the piece's copy; the rope table is the layer's forward table (YaRN: the
/// layer attends a stream). Both projections are read in the gemvs' layout,
/// a token per column: at more than one token, the row-major scratch's.
fn enqueue_indexer(
    cx: &Cx<'_>,
    lp: &LayerPlan,
    ip: &IndexerPlan,
    s: &mut Scratch,
    normed_act: NormedAct,
    joined: Joined,
    io: IndexerIo<'_>,
) -> Result<(), GpuError> {
    let (d, w, gpu, stream, m) = (cx.d, cx.w, cx.gpu, cx.gpu.stream(), cx.m);
    let refuse = |detail: &str| GpuError::Shape {
        what: WHAT,
        detail: detail.to_string(),
    };
    let kernels =
        cx.k.index
            .as_ref()
            .ok_or_else(|| refuse("an indexer layer on a piece that loaded no indexer"))?;
    let sel = s
        .select
        .as_mut()
        .ok_or_else(|| refuse("an indexer layer on a piece with no indexer scratch"))?;
    let Scratch {
        qkv,
        q,
        raw,
        acts,
        source,
        normed,
        q_a_normed,
        ..
    } = s;
    // Where the two projections sit in the gemvs' layout: the scratch's
    // parts on a piece of one token, the joins' places in the row-major
    // scratch otherwise.
    let q_rows = d.n_head * d.head_dim;
    let w_at = d.q_lora_rank + d.head_dim;
    let (w_rows, q_len) = (indexer::HEADS, indexer::HEADS * indexer::HEAD_DIM);
    let acts = count_of(acts, m)?;
    if !joined.query {
        let q_b = Dense::of(w, &ip.q_b, d.q_lora_rank, q_len)?;
        let q_act = q_b.reads_q8_1().then_some(&acts.q_a);
        match raw.as_mut() {
            None => {
                cx.k.dense
                    .enqueue_m(gpu, q_b, q_a_normed, q_act, m, q.part_mut(INDEX_Q))?
            }
            Some(r) => {
                let mut dst = rows_of_mut(&mut r.q, q_rows, q_len, m)?;
                cx.k.dense
                    .enqueue_m(gpu, q_b, q_a_normed, q_act, m, &mut dst)?;
            }
        }
    }
    if !joined.weights {
        let own = count_of(&mut sel.act, m)?;
        let act = match (lp.source.is_some(), source.as_ref(), normed_act) {
            (true, Some(src), _) => &src.act[m - 1],
            (_, _, NormedAct::Own) => &acts.normed,
            _ => {
                gpu.enqueue_quantize_q8_1_layer(normed, own, cx.layer)?;
                &*own
            }
        };
        match raw.as_mut() {
            None => gpu.enqueue_gemv_q3k(q3_k(w, &ip.proj)?, act, qkv.part_mut(INDEX_W))?,
            Some(r) => {
                let mut dst = rows_of_mut(&mut r.qkv, w_at, w_rows, m)?;
                gpu.enqueue_gemv_q3k(q3_k(w, &ip.proj)?, act, &mut dst)?;
            }
        }
    }
    let (q_in, w_in) = match raw.as_ref() {
        None => (
            view(q.part(INDEX_Q), 0, q_len * m)?,
            view(qkv.part(INDEX_W), 0, w_rows * m)?,
        ),
        Some(r) => (
            rows_of(&r.q, q_rows, q_len, m)?,
            rows_of(&r.qkv, w_at, w_rows, m)?,
        ),
    };
    let words = cx.words.view::<u32>(0, cx.words.layout.len)?;
    let scratch = count_of(&mut sel.streams, m)?[io.stream]
        .as_mut()
        .ok_or_else(|| refuse("an indexer layer's stream has no indexer scratch"))?;
    kernels.enqueue(
        stream,
        IndexerArgs {
            q: &q_in,
            w: &w_in,
            ints: &words,
            n_vis_at: cx.words.layout.streams[io.stream].nvis,
            top_k_at: cx.words.layout.top_k,
            tables: &words,
            rope_at: lp.forward,
            rope_stride: d.rope_dims,
            keys: io.keys,
            tokens: m,
            scratch,
            list: io.list,
            stride: io.top_k,
        },
    )?;
    sel.last = Some((m, io.stream));
    Ok(())
}

/// A compressor layer's own launches, after the norm left the q8_1 input:
/// the projections (kv and the gate in one launch when [`join_projections`]
/// joined them), the pooled (or ratio-1) row into the cache, then the index
/// key. At more than one token the projections reach the pooling
/// token-major through `raw`; so does the key whenever the geometry holds
/// more than one group slot, whatever the pass's token count.
fn enqueue_source(
    cx: &Cx<'_>,
    sp: &SourcePlan,
    src: &mut SourceScratch,
    raw: &mut Option<Raw>,
    io: &mut SourceIo<'_>,
) -> Result<(), GpuError> {
    let (d, w, gpu, stream, m) = (cx.d, cx.w, cx.gpu, cx.gpu.stream(), cx.m);
    let sw = &cx.words.layout.streams[sp.stream];
    let step = cx.words.view::<u32>(sw.step, sw.geom.words())?;
    let cs = cx
        .words
        .view::<f32>(sw.cs, sw.geom.max_groups * d.rope_dims)?;
    let gain = vector(w, &sp.norm)?;
    let missing = GpuError::State {
        what: WHAT,
        missing: "the compressor's row-major scratch",
    };
    let (mut raw_proj, raw_key) = match raw.as_mut() {
        Some(r) => (Some(r.proj.as_mut().ok_or(missing)?), r.key.as_mut()),
        None => (None, None),
    };
    let act = count_of(&mut src.act, m)?;
    let width = compress::WIDTH;
    match (&sp.gate, io.ring.as_mut()) {
        (Some(gate), Some((values, scores))) => {
            let joint = match joint_q3k(w, &sp.kv_gate)? {
                Some(t) if t.rows() == 2 * width => Some(t),
                Some(t) => {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "{} is {} rows; kv and the gate are {width} each",
                            sp.kv_gate,
                            t.rows(),
                        ),
                    });
                }
                None => None,
            };
            match (joint, raw_proj.as_deref_mut()) {
                (Some(t), None) => gpu.enqueue_gemv_q3k(t, act, src.proj.whole_mut())?,
                (Some(t), Some(r)) => gpu.enqueue_gemv_q3k(t, act, r)?,
                (None, None) => {
                    gpu.enqueue_gemv_q3k(q3_k(w, &sp.kv)?, act, src.proj.part_mut(COMP_KV))?;
                    gpu.enqueue_gemv_q3k(q3_k(w, gate)?, act, src.proj.part_mut(COMP_SCORE))?;
                }
                (None, Some(r)) => {
                    let mut kv = rows_of_mut(r, 0, width, m)?;
                    gpu.enqueue_gemv_q3k(q3_k(w, &sp.kv)?, act, &mut kv)?;
                    drop(kv);
                    let mut score = rows_of_mut(r, width, width, m)?;
                    gpu.enqueue_gemv_q3k(q3_k(w, gate)?, act, &mut score)?;
                }
            }
            if let Some(r) = raw_proj.as_deref() {
                for (part, r0) in [(COMP_KV, 0), (COMP_SCORE, width)] {
                    let rows = rows_of(r, r0, width, m)?;
                    cx.k.transpose
                        .enqueue(stream, &rows, width, m, src.proj.part_mut(part))?;
                }
            }
            cx.k.comp.enqueue_pool(
                stream,
                PoolArgs {
                    geom: sw.geom,
                    step: &step,
                    kv: src.proj.part(COMP_KV),
                    score: src.proj.part(COMP_SCORE),
                    gain,
                    cs: &cs,
                    eps: d.eps,
                    n_dims: d.rope_dims,
                    ring_kv: values.buf_mut(),
                    ring_score: scores.buf_mut(),
                    pre: &mut src.pre,
                    cache: &mut *io.rows,
                },
            )?;
        }
        (None, None) => {
            match raw_proj {
                None => {
                    gpu.enqueue_gemv_q3k(q3_k(w, &sp.kv)?, act, src.proj.part_mut(COMP_KV))?;
                }
                Some(r) => {
                    let mut kv = rows_of_mut(r, 0, width, m)?;
                    gpu.enqueue_gemv_q3k(q3_k(w, &sp.kv)?, act, &mut kv)?;
                    drop(kv);
                    let rows = rows_of(r, 0, width, m)?;
                    cx.k.transpose
                        .enqueue(stream, &rows, width, m, src.proj.part_mut(COMP_KV))?;
                }
            }
            cx.k.comp.enqueue_rows(
                stream,
                RowsArgs {
                    geom: sw.geom,
                    step: &step,
                    kv: src.proj.part(COMP_KV),
                    gain,
                    cs: &cs,
                    eps: d.eps,
                    n_dims: d.rope_dims,
                    pre: &mut src.pre,
                    cache: &mut *io.rows,
                },
            )?;
        }
        (gate, ring) => {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a compressor of ratio {} with a gate {} and a ring passed {}",
                    sw.geom.ratio,
                    gate.is_some(),
                    ring.is_some()
                ),
            });
        }
    }
    match (&sp.keys, io.keys.as_deref_mut()) {
        (Some((proj, norm)), Some(keys)) => {
            let groups = src.act_pre.m();
            gpu.enqueue_quantize_q8_1_layer(&src.pre, &mut src.act_pre, cx.layer)?;
            match raw_key {
                // The key's gemv runs the geometry's group slots, a column
                // each: the pooling wrote the pre-rope rows of the groups the
                // pass completes, and the index key reads only those.
                Some(rk) if groups > 1 => {
                    gpu.enqueue_gemv_q3k(q3_k(w, proj)?, &src.act_pre, rk)?;
                    cx.k.transpose
                        .enqueue(stream, rk, index_key::WIDTH, groups, &mut src.key)?;
                }
                _ => gpu.enqueue_gemv_q3k(q3_k(w, proj)?, &src.act_pre, &mut src.key)?,
            }
            cx.k.key.enqueue_index_key(
                stream,
                IndexKeyArgs {
                    geom: sw.geom,
                    step: &step,
                    k: &src.key,
                    gain: vector(w, norm)?,
                    cs: &cs,
                    eps: d.eps,
                    n_dims: d.rope_dims,
                    cache: keys,
                },
            )
        }
        (None, None) => Ok(()),
        (keys, passed) => Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a layer that owns index keys: {}; the caller passed a key cache: {}",
                keys.is_some(),
                passed.is_some()
            ),
        }),
    }
}

/// Layer `l` as the piece runs it: its names, its stream and its
/// compressor, checked against the plan and the file's layer table.
fn layer_plan(
    hp: &Hparams,
    planner: &Planner,
    words: &WordsLayout,
    l: usize,
) -> Result<LayerPlan, GpuError> {
    let refuse = |detail: String| GpuError::Shape {
        what: WHAT,
        detail: format!("layer {l}: {detail}"),
    };
    let kind = hp
        .layers
        .get(l)
        .ok_or_else(|| refuse(format!("the file has {} layers", hp.layers.len())))?;
    let stream = planner.layer_stream(l);
    if stream.is_some() != kind.stream.is_some() {
        return Err(refuse(format!(
            "the plan reads stream {stream:?}, the layer table {:?}",
            kind.stream
        )));
    }
    if let Some(st) = kind.stream {
        let reads = hp.layers.get(st.kv_source).and_then(|k| k.compressor);
        if reads.is_none() || planner.layer_stream(st.kv_source) != stream {
            return Err(refuse(format!(
                "its rows come from layer {}, which owns no compressor of its stream",
                st.kv_source
            )));
        }
    }
    let source = match (kind.compressor, kind.stream, stream) {
        (None, ..) if kind.index_keys => {
            return Err(refuse(
                "index keys without a compressor: the key projects the compressor's pre-rope row"
                    .to_string(),
            ));
        }
        (None, ..) => None,
        (Some(c), Some(st), Some(s)) if st.kv_source == l => {
            let ratio = words.streams[s].geom.ratio;
            if c.gated != (ratio > 1) {
                return Err(refuse(format!(
                    "a compressor of ratio {ratio} with a gate: {}",
                    c.gated
                )));
            }
            Some(SourcePlan {
                stream: s,
                kv: names::attn_compressor_kv(l),
                gate: c.gated.then(|| names::attn_compressor_gate(l)),
                kv_gate: joint_name(l, JOINT_KV_GATE),
                norm: names::attn_compressor_norm(l),
                keys: kind
                    .index_keys
                    .then(|| (names::indexer_attn_k(l), names::indexer_k_norm(l))),
            })
        }
        (Some(_), st, _) => {
            return Err(refuse(format!(
                "owns a compressor but attends {st:?}: a compressor writes its own layer's stream"
            )));
        }
    };
    // Every layer of a stream reads the list its top-k source wrote over the
    // same stream; an indexer layer is its own top-k source and scores the
    // keys of a layer of its stream that owns them.
    let of_stream = |src: usize| planner.layer_stream(src) == stream && src <= l;
    if let Some(st) = kind.stream {
        let ts = st.topk_source;
        if !(of_stream(ts) && hp.layers.get(ts).is_some_and(|k| k.indexer)) {
            return Err(refuse(format!(
                "its list comes from layer {ts}, which runs no indexer over its stream before it"
            )));
        }
    }
    let indexer = match kind.stream {
        Some(st) if kind.indexer => {
            let ks = st.index_key_source;
            if st.topk_source != l
                || !(of_stream(ks) && hp.layers.get(ks).is_some_and(|k| k.index_keys))
            {
                return Err(refuse(format!(
                    "runs the indexer as top-k source {} over the keys of layer {ks}, which owns \
                     none of its stream",
                    st.topk_source
                )));
            }
            if ks == l && source.as_ref().is_none_or(|s| s.keys.is_none()) {
                return Err(refuse(
                    "scores its own index keys and writes none".to_string(),
                ));
            }
            Some(IndexerPlan {
                q_b: names::indexer_attn_q_b(l),
                proj: names::indexer_proj(l),
                owns_keys: ks == l,
            })
        }
        None if kind.indexer => {
            return Err(refuse("runs the indexer and attends no stream".to_string()));
        }
        _ => None,
    };
    let t = |table| words.tables[table_index(table)];
    let (forward, back) = match stream {
        Some(_) => (t(Table::YarnForward), t(Table::YarnBack)),
        None => (t(Table::WindowForward), t(Table::WindowBack)),
    };
    Ok(LayerPlan {
        names: Names {
            hc_fn: names::hc_attn_fn(l),
            hc_scale: names::hc_attn_scale(l),
            hc_base: names::hc_attn_base(l),
            norm: names::attn_norm(l),
            q_a: names::attn_q_a(l),
            q_a_norm: names::attn_q_a_norm(l),
            q_b: names::attn_q_b(l),
            kv: names::attn_kv(l),
            kv_norm: names::attn_kv_a_norm(l),
            sinks: names::attn_sinks(l),
            out_a: names::attn_output_a(l),
            out_b: names::attn_output_b(l),
            qkv: joint_name(l, JOINT_QKV),
            q: joint_name(l, JOINT_Q),
        },
        stream,
        source,
        indexer,
        forward,
        back,
    })
}

/// The derived names [`join_projections`] files layer `l`'s row joins
/// under: `derived.blk.<l>.<what>`.
fn joint_name(l: usize, what: &str) -> String {
    format!("derived.blk.{l}.{what}")
}

/// q_a, kv and, on an indexer layer, the indexer's weights projection.
const JOINT_QKV: &str = "attn_q_a+kv+indexer.proj";
/// q_b and, on an indexer layer, the indexer's query projection.
const JOINT_Q: &str = "attn_q_b+indexer.attn_q_b";
/// A gated compressor's kv and gate projections.
const JOINT_KV_GATE: &str = "attn_compressor_kv+gate";

/// One row join [`join_projections`] makes on a layer: the derived name it
/// is filed under and its parts in row order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinGroup {
    /// The derived weight's name (`derived.blk.<l>.<what>`).
    pub joint: String,
    /// The file tensors whose rows the join concatenates, in that order.
    pub parts: Vec<String>,
}

/// The row joins layer `l` of the model `hp` describes would carry — the
/// one owner of which projections read one activation together:
///
/// - q_a, kv and, on an indexer layer, the indexer's weights projection
///   (the normed input);
/// - on an indexer layer, q_b and the indexer's query projection (q_a's
///   norm);
/// - a gated compressor's kv and gate projections (its normed input).
///
/// [`join_projections`] joins a group when every part is resident as Q3_K;
/// a gate that reads the plan against the resident weights folds the same
/// groups.
pub fn join_groups(hp: &Hparams, l: usize) -> Result<Vec<JoinGroup>, GpuError> {
    let kind = hp.layers.get(l).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("layer {l} of a file of {} layers", hp.layers.len()),
    })?;
    let indexer = kind.indexer && kind.stream.is_some();
    let owns_gated =
        kind.compressor.is_some_and(|c| c.gated) && kind.stream.is_some_and(|st| st.kv_source == l);
    let mut groups = Vec::with_capacity(3);
    let mut qkv = vec![names::attn_q_a(l), names::attn_kv(l)];
    if indexer {
        qkv.push(names::indexer_proj(l));
        groups.push(JoinGroup {
            joint: joint_name(l, JOINT_Q),
            parts: vec![names::attn_q_b(l), names::indexer_attn_q_b(l)],
        });
    }
    groups.push(JoinGroup {
        joint: joint_name(l, JOINT_QKV),
        parts: qkv,
    });
    if owns_gated {
        groups.push(JoinGroup {
            joint: joint_name(l, JOINT_KV_GATE),
            parts: vec![names::attn_compressor_kv(l), names::attn_compressor_gate(l)],
        });
    }
    Ok(groups)
}

/// For layers `layers` of the model `hp` describes, move each group of
/// projections that read the same q8_1 activation ([`join_groups`]) into one
/// row stream ([`Weights::join_rows`]), so the chain runs the group as one
/// `q3k_gemv` launch — each row the same kernel on the same bits, only its
/// index moved. A group is joined only when every member is resident as
/// Q3_K; any other format keeps its separate launches. Load-time only.
pub fn join_projections(
    stream: &CudaStream,
    hp: &Hparams,
    layers: Range<usize>,
    w: &mut Weights,
) -> Result<(), GpuError> {
    let is_q3k = |w: &Weights, name: &str| {
        matches!(
            w.get(name),
            Some(DevWeight::KQuant {
                ty: GgmlType::Q3_K,
                ..
            })
        )
    };
    for l in layers {
        for g in join_groups(hp, l)? {
            if g.parts.iter().all(|p| is_q3k(w, p)) {
                let parts: Vec<&str> = g.parts.iter().map(String::as_str).collect();
                w.join_rows(stream, &parts, g.joint)?;
            }
        }
    }
    Ok(())
}

/// A layer's projections of the normed input as resident: one joined Q3_K
/// row stream, or q_a and kv in their own formats.
#[derive(Clone, Copy)]
enum Qkv<'w> {
    Joint(&'w DeviceTensor<u32>),
    Parts { q_a: Dense<'w>, kv: Dense<'w> },
}

impl<'w> Qkv<'w> {
    /// Layer `lp`'s: the join when resident — refused unless its rows are
    /// q_a's, kv's and, on an indexer layer, the indexer weights' — else the
    /// two projections.
    fn of(w: &'w Weights, lp: &LayerPlan, d: &Dims) -> Result<Qkv<'w>, GpuError> {
        let n = &lp.names;
        let index = if lp.indexer.is_some() {
            indexer::HEADS
        } else {
            0
        };
        match joint_q3k(w, &n.qkv)? {
            Some(t) if t.rows() == d.q_lora_rank + d.head_dim + index => Ok(Qkv::Joint(t)),
            Some(t) => Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} is {} rows; q_a, kv and the indexer's weights are {}, {} and {index}",
                    n.qkv,
                    t.rows(),
                    d.q_lora_rank,
                    d.head_dim
                ),
            }),
            None => Ok(Qkv::Parts {
                q_a: Dense::of(w, &n.q_a, d.n_embd, d.q_lora_rank)?,
                kv: Dense::of(w, &n.kv, d.n_embd, d.head_dim)?,
            }),
        }
    }

    /// Whether the projections read the normed input in q8_1.
    fn reads_q8_1(&self) -> bool {
        match self {
            Qkv::Joint(_) => true,
            Qkv::Parts { q_a, kv } => q_a.reads_q8_1() || kv.reads_q8_1(),
        }
    }
}

/// A layer's projections of q_a's norm as resident: one joined Q3_K row
/// stream, or q_b and, on an indexer layer, the indexer's query in their own
/// formats.
#[derive(Clone, Copy)]
enum Query<'w> {
    Joint(&'w DeviceTensor<u32>),
    Parts {
        q_b: Dense<'w>,
        index_q: Option<Dense<'w>>,
    },
}

impl<'w> Query<'w> {
    /// Layer `lp`'s: the join when resident — refused unless its rows are
    /// q_b's and the indexer query's — else the projections.
    fn of(w: &'w Weights, lp: &LayerPlan, d: &Dims) -> Result<Query<'w>, GpuError> {
        let n = &lp.names;
        let q_rows = d.n_head * d.head_dim;
        let index_rows = indexer::HEADS * indexer::HEAD_DIM;
        match (joint_q3k(w, &n.q)?, &lp.indexer) {
            (Some(t), Some(_)) if t.rows() == q_rows + index_rows => Ok(Query::Joint(t)),
            (Some(t), _) => Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} is {} rows on a layer that runs the indexer: {}; q_b and the indexer's \
                     query are {q_rows} and {index_rows}",
                    n.q,
                    t.rows(),
                    lp.indexer.is_some()
                ),
            }),
            (None, ip) => Ok(Query::Parts {
                q_b: Dense::of(w, &n.q_b, d.q_lora_rank, q_rows)?,
                index_q: ip
                    .as_ref()
                    .map(|ip| Dense::of(w, &ip.q_b, d.q_lora_rank, index_rows))
                    .transpose()?,
            }),
        }
    }

    /// Whether q_a's norm leaves its q8_1 form for them.
    fn reads_q8_1(&self) -> bool {
        match self {
            Query::Joint(_) => true,
            Query::Parts { q_b, index_q } => {
                q_b.reads_q8_1() || index_q.is_some_and(|q| q.reads_q8_1())
            }
        }
    }
}

/// The derived row join `name` when [`join_projections`] filed it; refused
/// when the name holds anything but a Q3_K row stream.
fn joint_q3k<'w>(w: &'w Weights, name: &str) -> Result<Option<&'w DeviceTensor<u32>>, GpuError> {
    match w.get(name) {
        None => Ok(None),
        Some(DevWeight::KQuant {
            ty: GgmlType::Q3_K,
            w,
            ..
        }) => Ok(Some(w)),
        Some(_) => Err(missing(name, "a Q3_K row join")),
    }
}

/// Q3_K weight `name`'s row stream.
fn q3_k<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant {
            ty: GgmlType::Q3_K,
            w,
            ..
        }) => Ok(w),
        _ => Err(missing(name, "resident as Q3_K")),
    }
}

/// F32 vector `name`: a gain, the sinks, HC_PRE's scales or offsets.
fn vector<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w, .. }) => Ok(w.buf()),
        _ => Err(missing(name, "resident as F32")),
    }
}

fn missing(name: &str, need: &'static str) -> GpuError {
    GpuError::Tensor {
        what: WHAT,
        name: name.to_string(),
        need,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::StepInts;
    use crate::params::{ImageDims, StepImage};
    use crate::rope::RopeSpec;
    use model::arch::deepseek41::plan::StepPlan;

    const WINDOW: u32 = 128;
    const CTX: u64 = 4096;
    const ROPE_DIMS: usize = 64;

    /// The gather's pairs applied on the host to the image of every position
    /// through two windows and past the ratios' residues give, per stream,
    /// the attention's pair `[window length, visible rows]` and exactly the
    /// words `CompGeom::pack` writes from the same plan; the window-only pair
    /// and the tables are the image's.
    #[test]
    fn gathered_words_are_the_launch_layouts() {
        let ratios = [0, 0, 2, 2, 1, 1];
        let planner = Planner::new(WINDOW, &ratios, 4, CTX).expect("a planner");
        let layout = ImageLayout::new(ImageDims {
            tokens: 1,
            window: WINDOW,
            stream_ratios: planner.stream_ratios().to_vec(),
            rope_dims: ROPE_DIMS,
            n_embd: 8,
            embd_bytes: 16,
            engram_bytes: 6,
        })
        .expect("a layout");
        let (words, src, dst) = plan_words(&layout, CTX as usize).expect("the pairs");
        let window = RopeSpec::window(10_000.0, ROPE_DIMS);
        let yarn = RopeSpec::yarn(160_000.0, 16.0, 65_536, 32.0, 1.0, ROPE_DIMS);
        let mut image = StepImage::new(layout.clone(), &window, &yarn).expect("an image");
        let mut plan = StepPlan::default();
        let history: Vec<u32> = (0..600).map(|t| t * 7 % 1000).collect();
        for pos in [0u32, 1, 2, 3, 4, 126, 127, 128, 129, 300, 301, 511, 512] {
            let at = pos as usize;
            planner
                .plan_into(&history[at..=at], pos, &history[..at], &mut plan)
                .expect("a plan");
            image
                .build(&plan, &[0u8; 16], &[0u8; 6])
                .expect("an image of the plan");
            let mut got = vec![0u32; words.len];
            for (&s, &d) in src.iter().zip(&dst) {
                got[d as usize] = image.words()[s as usize];
            }
            let len = pos.min(WINDOW - 1) + 1;
            assert_eq!(got[words.pos], pos, "position at {pos}");
            assert_eq!(
                got[words.window_vis..][..2],
                [len, 0],
                "window pair at {pos}"
            );
            let img = layout.view(image.words()).expect("a view");
            for (t, &at) in Table::ALL.iter().zip(&words.tables) {
                assert_eq!(&got[at..][..ROPE_DIMS], img.table(0, *t), "{t:?} at {pos}");
            }
            for (s, (sw, st)) in words.streams.iter().zip(&plan.streams).enumerate() {
                assert_eq!(
                    got[sw.vis..][..2],
                    [len, st.n_visible[0]],
                    "stream {s} pair at {pos}"
                );
                let mut packed = vec![0u32; sw.geom.words()];
                sw.geom
                    .pack(
                        &StepInts {
                            write_row: &st.state_write,
                            read: &st.state_read,
                            persist_src: &st.persist_src,
                            persist_dst: &st.persist_dst,
                        },
                        &mut packed,
                    )
                    .expect("the plan packs");
                assert_eq!(
                    &got[sw.step..][..packed.len()],
                    &packed[..],
                    "stream {s} step words at {pos}"
                );
                let cs = &got[sw.cs..][..ROPE_DIMS];
                match img.row_table(s, 0) {
                    Some(t) => assert_eq!(cs, t, "stream {s} row table at {pos}"),
                    None => assert_eq!(
                        cs,
                        img.table(0, Table::YarnForward),
                        "stream {s} ratio-1 table at {pos}"
                    ),
                }
            }
        }
    }
}
