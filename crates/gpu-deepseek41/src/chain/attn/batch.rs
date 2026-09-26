//! A prompt batch's attention sub-layer with its projections batch-wide
//! ([`AttnChain::enqueue_batch_layer`]; its buffers are an [`AttnBatch`]).
//!
//! The layer's chunks run in position order, as
//! [`AttnChain::enqueue_layer_staged`] and [`AttnChain::enqueue_layer_part`]
//! run them one chunk at a time, but a launch that reads the layer's input
//! alone runs once over a sub-block of chunks instead of once a chunk:
//!
//! 1. before the chunks, once over the layer's block: HC_PRE on the piece's
//!    branch, in groups of [`HC_MAX_TOKENS`] tokens in one grid
//!    ([`HcKernels::enqueue_pre_groups`]);
//! 2. per sub-block, before its chunks: the attention norm with its q8_1
//!    form, the compressor's projections, the joined q_a·kv (·indexer
//!    weights) projection and, in the block, q_a's norm, the query
//!    projection and the tail rope — each gemv one launch over the
//!    sub-block's chunks as column groups ([`ColGroups`], a chunk a group),
//!    each copy token-major one launch;
//! 3. per chunk, in position order — the stages that carry state from a
//!    position to the next: the compressor's pooled row and index key, the
//!    latent row into the staging (before the block, into the ring), the
//!    indexer, the staged attention and its commit;
//! 4. per sub-block of the block, after its chunks: the inverse rope, the
//!    heads' q8_1 form, wo_a's block diagonal, wo_a's q8_1 form, wo_b and
//!    its copy token-major into the layer's output;
//! 5. after the chunks, once over the block: the branch's join and HC_POST
//!    with the next fold.
//!
//! Every launch writes, per token, what the chunk's own launch writes: a
//! column group is the m-column launch on its columns, bit for bit
//! (`cores::q3k_row_dot`'s contract), the norms and the quantizer are
//! column-local, HC_PRE, the ropes and HC_POST token-local. The projections'
//! tensors are resolved once a layer-batch ([`LayerTensors`]).
//!
//! A sub-block is a chunk shorter than [`HC_MAX_TOKENS`] alone, or a run of
//! 1, 2, 4, 8 or 16 whole chunks ([`SubBlocks`]): the fused norm writes the
//! q8_1 form of exactly its tokens, so the batch keeps one per token count a
//! sub-block can have ([`SUB_COUNTS`]).

use std::ops::Range;

use bloomery_gpu::{ColGroups, FaultSink, Gpu, GpuError, Q8Act};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, sys};

use super::*;
use crate::dense::{Q3kHeadsGroupsArgs, RowsPart};
use crate::params::ImageLayout;
use crate::span::{span, span_mut};

const WHAT_BATCH: &str = "deepseek41 AttnBatch";

/// Whole chunks a sub-block holds at most.
const SUB_CHUNKS: usize = 16;

/// Tokens a sub-block holds at most.
pub const SUB_TOKENS: usize = SUB_CHUNKS * HC_MAX_TOKENS;

/// A sub-block's token counts, ascending: a short chunk's, a whole chunk's,
/// and runs of 2, 4, 8 and 16 whole chunks.
const SUB_COUNTS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128];
const _: () = assert!(SUB_COUNTS[SUB_COUNTS.len() - 1] == SUB_TOKENS);
const _: () = assert!(SUB_COUNTS[HC_MAX_TOKENS - 1] == HC_MAX_TOKENS);

/// Events [`AttnBatch`] records around a layer's projection phases at most
/// when timed: a start and an end for each sub-block's phase before its
/// chunks and after them, over a latent part and a block of at most
/// [`sub_block_bound`] sub-blocks each.
fn phase_events(chunks: usize) -> usize {
    2 * 3 * sub_block_bound(chunks)
}

/// The most sub-blocks [`SubBlocks`] cuts `chunks` chunks into: a short
/// chunk at either end, the runs of [`SUB_CHUNKS`], and one run of each
/// smaller power of two.
fn sub_block_bound(chunks: usize) -> usize {
    2 + chunks / SUB_CHUNKS + SUB_CHUNKS.trailing_zeros() as usize
}

/// The chunks `k .. end` of `cuts` as sub-blocks, in order: a chunk shorter
/// than [`HC_MAX_TOKENS`] alone (only a batch's first and last chunk can
/// be), and each run of whole chunks as runs of [`SUB_CHUNKS`], then of 8,
/// 4, 2 and 1 — so a sub-block's token count is one of [`SUB_COUNTS`].
struct SubBlocks<'a> {
    cuts: &'a [Range<usize>],
    k: usize,
    end: usize,
}

impl<'a> SubBlocks<'a> {
    fn new(cuts: &'a [Range<usize>], k: usize, end: usize) -> SubBlocks<'a> {
        SubBlocks { cuts, k, end }
    }
}

impl Iterator for SubBlocks<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        let k = self.k;
        if k >= self.end {
            return None;
        }
        let whole = |j: usize| self.cuts[j].len() == HC_MAX_TOKENS;
        let n = if whole(k) {
            let run = (k..self.end)
                .take_while(|&j| whole(j))
                .count()
                .min(SUB_CHUNKS);
            1 << (usize::BITS - 1 - run.leading_zeros())
        } else {
            1
        };
        self.k = k + n;
        Some(k..k + n)
    }
}

/// `n`'s place in [`SUB_COUNTS`], refused for a count no sub-block has or
/// one past the `held` counts a batch holds.
fn count_at(n: usize, held: usize) -> Result<usize, GpuError> {
    SUB_COUNTS
        .iter()
        .position(|&c| c == n)
        .filter(|&i| i < held)
        .ok_or_else(|| GpuError::Shape {
            what: WHAT_BATCH,
            detail: format!(
                "a sub-block of {n} tokens; the batch holds q8_1 forms of {:?} tokens",
                &SUB_COUNTS[..held.min(SUB_COUNTS.len())]
            ),
        })
}

/// A layer's caches as one chunk of a prompt batch reads and writes them:
/// its window ring and shadow, its compressed rows, and the chunk's list.
pub struct ChunkCaches<'a> {
    pub ring: &'a mut DeviceTensor<u16>,
    pub shadow: &'a mut DeviceTensor<u16>,
    pub compressed: Compressed<'a>,
    pub selection: Selection<'a>,
}

/// Where [`AttnChain::enqueue_batch_layer`] finds chunk `k`'s caches: the
/// layer's, with the chunk's list.
pub trait ChunkSource {
    fn chunk(&mut self, k: usize) -> Result<ChunkCaches<'_>, GpuError>;
}

/// What a layer of a prompt batch reads and writes besides its caches.
pub struct BatchIo<'a> {
    /// The batch's chunks, positions cut at multiples of [`HC_MAX_TOKENS`],
    /// chunk `k` reading row `k` of the piece's words: the layer runs its
    /// latent rows from chunk `run` on and its block from chunk `full` on.
    pub cuts: &'a [Range<usize>],
    pub run: usize,
    pub full: usize,
    /// Per token of the batch: the streams and the fold the sub-layer reads,
    /// and the streams and fold it leaves over the block's tokens.
    pub streams_in: &'a DeviceBuffer<f32>,
    pub fold_in: &'a DeviceBuffer<f32>,
    pub streams_out: &'a mut DeviceBuffer<f32>,
    pub fold_out: &'a mut DeviceBuffer<f32>,
    /// A chunk's latent rows before its commit ([`StageIo::rows`]).
    pub staging: &'a mut DeviceTensor<u16>,
}

/// A layer's tensors as the batch path reads them, resolved once a
/// layer-batch. The path runs the joined Q3_K formats alone and refuses any
/// other by name.
struct LayerTensors<'w> {
    hc: HcParams<'w>,
    norm: &'w DeviceBuffer<f32>,
    /// q_a, kv and, on an indexer layer, the indexer's weights, joined.
    qkv: &'w DeviceTensor<u32>,
    q_a_norm: &'w DeviceBuffer<f32>,
    /// q_b — joined with the indexer's query on an indexer layer — and its
    /// rows.
    query: &'w DeviceTensor<u32>,
    kv_norm: &'w DeviceBuffer<f32>,
    sinks: &'w DeviceBuffer<f32>,
    out_a: &'w DeviceTensor<u32>,
    out_b: &'w DeviceTensor<u32>,
    source: Option<SourceTensors<'w>>,
}

impl<'w> LayerTensors<'w> {
    fn of(
        w: &'w Weights,
        lp: &LayerPlan,
        d: &Dims,
        words: &WordsLayout,
    ) -> Result<LayerTensors<'w>, GpuError> {
        let n = &lp.names;
        let refuse = |name: &str, need: &'static str| GpuError::Tensor {
            what: WHAT_BATCH,
            name: name.to_string(),
            need,
        };
        let qkv = match Qkv::of(w, lp, d)? {
            Qkv::Joint(t) => t,
            Qkv::Parts { .. } => return Err(refuse(&n.qkv, "the joined Q3_K rows")),
        };
        let query = match (Query::of(w, lp, d)?, &lp.indexer) {
            (Query::Joint(t), _) => t,
            (
                Query::Parts {
                    q_b: Dense::Q3K(t),
                    index_q: None,
                },
                None,
            ) => t,
            _ => return Err(refuse(&n.q, "Q3_K, joined with the indexer's query")),
        };
        let groups = d.n_head * d.head_dim / d.group_k;
        let out_a = match Dense::of(w, &n.out_a, d.group_k, groups * d.o_lora_rank)? {
            Dense::Q3K(t) => t,
            _ => return Err(refuse(&n.out_a, "Q3_K")),
        };
        let out_b = match Dense::of(w, &n.out_b, groups * d.o_lora_rank, d.n_embd)? {
            Dense::Q3K(t) => t,
            _ => return Err(refuse(&n.out_b, "Q3_K")),
        };
        let source = match &lp.source {
            Some(sp) => {
                let ratio = words.streams[sp.stream].geom.ratio;
                Some(SourceTensors::of(w, sp, ratio, ratio > 1)?)
            }
            None => None,
        };
        Ok(LayerTensors {
            hc: HcParams {
                w: q3_k(w, &n.hc_fn)?,
                scale: vector(w, &n.hc_scale)?,
                base: vector(w, &n.hc_base)?,
                eps: d.hc_eps,
                iters: d.hc_iters,
            },
            norm: vector(w, &n.norm)?,
            qkv,
            q_a_norm: vector(w, &n.q_a_norm)?,
            query,
            kv_norm: vector(w, &n.kv_norm)?,
            sinks: vector(w, &n.sinks)?,
            out_a,
            out_b,
            source,
        })
    }
}

/// The buffers of [`AttnChain::enqueue_batch_layer`] for batches of up to
/// `tokens` positions ([`AttnChain::batch`]): the launches before a
/// sub-block's chunks write them, the chunks and the launches after them
/// read them. Per token of the batch: HC_PRE's result, the sub-layer's
/// output and the rope tables; per token of a sub-block
/// ([`SUB_TOKENS`] at most): everything else.
pub struct AttnBatch {
    tokens: usize,
    /// Tokens a sub-block's buffers hold.
    sub: usize,
    rope_dims: usize,
    hc_pre: HcPreScratch,
    mixes: DeviceBuffer<f32>,
    hc: DeviceBuffer<f32>,
    /// The attention norm, f32, and its q8_1 form per [`SUB_COUNTS`] count.
    normed: DeviceBuffer<f32>,
    normed_acts: Vec<Q8Act>,
    /// The joined projection of the normed input, as the column groups lay
    /// it out ([`Gpu::enqueue_gemv_q3k_groups`]), then q_a and the latent
    /// projection token-major.
    raw_qkv: DeviceBuffer<f32>,
    q_a: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    /// q_a's norm, f32, and its q8_1 form per count.
    q_a_normed: DeviceBuffer<f32>,
    q_a_acts: Vec<Q8Act>,
    /// q_a's norm's projections as laid out, then the query heads
    /// token-major after the tail rope.
    raw_q: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    /// On a card with a compressor layer: its projections as laid out — the
    /// gate's from a sub-block's rows of kv on when they are two launches —
    /// then kv ([`COMP_KV`]) and the gate's ([`COMP_SCORE`]) token-major,
    /// each [`HC_MAX_TOKENS`] rows past a sub-block's: the pooling launchers
    /// bound a chunk's reads by the stream geometry's tokens.
    comp_raw: Option<DeviceBuffer<f32>>,
    comp: Option<PartedBuffer<f32, 2>>,
    /// The attention output, token-major.
    y: DeviceBuffer<f32>,
    /// The heads' q8_1 form, a column per output group and token; wo_a
    /// token-major; its q8_1 form; wo_b as laid out.
    heads: Q8Act,
    wo_a: DeviceBuffer<f32>,
    wo_a_act: Q8Act,
    raw_out: DeviceBuffer<f32>,
    /// The sub-layer's output, per token of the batch: HC_POST's input.
    out: DeviceBuffer<f32>,
    /// Per table of [`Table::ALL`], per token of the batch, its rope table
    /// ([`AttnBatch::stage_tables`]), and the host copy it is uploaded from.
    tables: DeviceBuffer<f32>,
    tables_host: Vec<f32>,
    /// Queue entries enqueued since the last [`AttnBatch::take_entries`].
    entries: u64,
    /// With card timing on: an event pair around each projection phase, the
    /// pairs recorded since the last [`AttnBatch::take_card_ms`].
    marks: Vec<CudaEvent>,
    marked: usize,
    timed: bool,
}

/// The lengths of an [`AttnBatch`]'s buffers, from which it is allocated and
/// its bytes are counted before the allocation.
struct BatchLens {
    /// The f32 buffers' values, in [`AttnBatch`]'s field order.
    f32s: [usize; 16],
    /// Q8_1 forms: columns and values a column.
    acts: Vec<(usize, usize)>,
    hc_groups: usize,
    hc_k: usize,
}

impl BatchLens {
    fn bytes(&self) -> usize {
        let hc_pieces = self.hc_k / HC_PIECE;
        4 * self.f32s.iter().sum::<usize>()
            + self
                .acts
                .iter()
                .map(|&(m, k)| q8act_bytes(m, k))
                .sum::<usize>()
            + 4 * (HC_MIX * hc_pieces * HC_MAX_TOKENS * self.hc_groups
                + hc_pieces * HC_MAX_TOKENS * self.hc_groups
                + self.hc_groups)
    }
}

/// A q8_1 form of `m` columns of `k` values: past eight columns only the
/// per-slot allocation takes them.
fn act(stream: &CudaStream, m: usize, k: usize) -> Result<Q8Act, GpuError> {
    if m <= 8 {
        Q8Act::with_k(stream, m, k)
    } else {
        Q8Act::with_slots(stream, m, k)
    }
}

impl AttnBatch {
    /// Device bytes of the buffers.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let f32s = [
            &self.mixes,
            &self.hc,
            &self.normed,
            &self.raw_qkv,
            &self.q_a,
            &self.kv,
            &self.q_a_normed,
            &self.raw_q,
            &self.q,
            &self.y,
            &self.wo_a,
            &self.raw_out,
            &self.out,
            &self.tables,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.comp_raw.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.comp.as_ref().map_or(0, |c| c.whole().num_bytes());
        let acts = self
            .normed_acts
            .iter()
            .chain(&self.q_a_acts)
            .chain([&self.heads, &self.wo_a_act])
            .map(|a| q8act_bytes(a.m(), 256 * a.n_sb()))
            .sum::<usize>();
        f32s + acts + self.hc_pre.device_bytes()
    }

    /// Chunk `m` tokens' rope tables into the host copy from token `at` on:
    /// token `t`'s tables of the chunk image `words` (of `layout`) as
    /// [`ImageView::table`](crate::params::ImageView::table) reads them.
    /// Refused past the batch's tokens or the image's.
    pub fn stage_tables(
        &mut self,
        layout: &ImageLayout,
        words: &[u32],
        at: usize,
        m: usize,
    ) -> Result<(), GpuError> {
        let img = layout.view(words)?;
        let nd = self.rope_dims;
        if at + m > self.tokens || m > layout.dims().tokens || layout.dims().rope_dims != nd {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "{m} tokens' tables from token {at} of a batch of {}, from an image of {} \
                     tokens of {} rope values (the batch's {nd})",
                    self.tokens,
                    layout.dims().tokens,
                    layout.dims().rope_dims
                ),
            });
        }
        for (i, table) in Table::ALL.into_iter().enumerate() {
            for t in 0..m {
                let dst = (i * self.tokens + at + t) * nd;
                for (d, &w) in self.tables_host[dst..dst + nd]
                    .iter_mut()
                    .zip(img.table(t, table))
                {
                    *d = f32::from_bits(w);
                }
            }
        }
        Ok(())
    }

    /// The host copy of the tables to the card, one transfer; it waits for
    /// the stream.
    pub fn upload_tables(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.tables.copy_from_host(stream, &self.tables_host)?;
        Ok(())
    }

    /// Queue entries — launches, event records and stream waits — the batch
    /// path enqueued since the last call, and zero again.
    pub fn take_entries(&mut self) -> u64 {
        std::mem::take(&mut self.entries)
    }

    /// Time each layer's projection phases with events from the next layer
    /// on: an event pair around each ([`AttnBatch::take_card_ms`]). The
    /// events are made the first time it is turned on, for batches of the
    /// piece's rows of chunks over `layers` layers.
    pub fn set_card_timing(&mut self, gpu: &Gpu, on: bool, layers: usize) -> Result<(), GpuError> {
        let want = layers * phase_events(self.tokens.div_ceil(HC_MAX_TOKENS) + 1);
        while on && self.marks.len() < want {
            self.marks.push(
                gpu.context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            );
        }
        self.timed = on;
        self.marked = 0;
        Ok(())
    }

    /// The card time of the projection phases timed since the last call, in
    /// ms, summed — which waits for the last of them — and none again.
    pub fn take_card_ms(&mut self) -> Result<f64, GpuError> {
        let (pairs, _) = self.marks[..self.marked].as_chunks::<2>();
        let mut ms = 0.0f64;
        for [e0, e1] in pairs {
            ms += f64::from(e0.elapsed_ms(e1)?);
        }
        self.marked = 0;
        Ok(ms)
    }

    /// With card timing on, the next event recorded on `stream` — a phase's
    /// start or end; the pool is sized for every phase a batch can have, and
    /// running past it is refused.
    fn mark(&mut self, stream: &CudaStream, entries: &mut u64) -> Result<(), GpuError> {
        if !self.timed {
            return Ok(());
        }
        let e = self.marks.get(self.marked).ok_or(GpuError::State {
            what: WHAT_BATCH,
            missing: "an event for a projection phase: the pool is sized for a batch",
        })?;
        e.record(stream)?;
        self.marked += 1;
        *entries += 1;
        Ok(())
    }
}

/// What every launch of one layer-batch reads besides its buffers.
struct BatchCx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    d: &'a Dims,
    words: &'a Words,
    lp: &'a LayerPlan,
    t: &'a LayerTensors<'a>,
    layer: usize,
    fault: FaultSink,
    cuts: &'a [Range<usize>],
    /// The batch's first position.
    base: usize,
    top_k: usize,
}

impl BatchCx<'_> {
    /// The batch's token chunk `k` starts at, its end past the last chunk.
    fn token(&self, k: usize) -> usize {
        match self.cuts.get(k) {
            Some(r) => r.start - self.base,
            None => self.cuts.last().map_or(0, |r| r.end - self.base),
        }
    }

    /// Chunk `k`'s launch context: its row of words and its tokens.
    fn chunk(&self, k: usize) -> Result<Cx<'_>, GpuError> {
        Ok(Cx {
            gpu: self.gpu,
            w: self.w,
            k: self.k,
            d: self.d,
            words: self.words.of(k)?,
            layer: self.layer,
            fault: self.fault,
            m: self.cuts[k].len(),
        })
    }

    /// The column groups of sub-block `sb`'s tokens: a chunk a group.
    fn groups(&self, sb: &Range<usize>) -> Result<ColGroups, GpuError> {
        ColGroups::new(
            0,
            self.cuts[sb.start].len(),
            self.token(sb.end) - self.token(sb.start),
        )
    }

    /// Rows of the joined projection of the normed input, and of q_a's norm.
    fn qkv_rows(&self) -> usize {
        self.t.qkv.rows()
    }

    fn q_rows(&self) -> usize {
        self.t.query.rows()
    }
}

impl AttnChain {
    /// The buffers of [`AttnChain::enqueue_batch_layer`] for batches of up to
    /// `tokens` positions, refused by name when they do not fit the card's
    /// free memory. Load-time only.
    pub fn batch(&self, gpu: &Gpu, tokens: usize) -> Result<AttnBatch, GpuError> {
        let d = &self.dims;
        let stream = gpu.stream();
        let s = SUB_TOKENS.min(tokens.next_multiple_of(HC_MAX_TOKENS));
        let any_source = self.plans.iter().any(|p| p.source.is_some());
        let any_indexer = self.plans.iter().any(|p| p.indexer.is_some());
        let index_w = if any_indexer { indexer::HEADS } else { 0 };
        let index_q = if any_indexer {
            indexer::HEADS * indexer::HEAD_DIM
        } else {
            0
        };
        let q_rows = d.n_head * d.head_dim;
        let groups = q_rows / d.group_k;
        let width = compress::WIDTH;
        let comp_raw = if any_source { 2 * s * width } else { 0 };
        let comp = if any_source {
            (s + HC_MAX_TOKENS) * width
        } else {
            0
        };
        let nd = d.rope_dims;
        let counts: Vec<usize> = SUB_COUNTS.iter().copied().filter(|&c| c <= s).collect();
        let lens = BatchLens {
            f32s: [
                tokens * HC_MIX,
                tokens * HC_MIX,
                s * d.n_embd,
                s * (d.q_lora_rank + d.head_dim + index_w),
                s * d.q_lora_rank,
                s * d.head_dim,
                s * d.q_lora_rank,
                s * (q_rows + index_q),
                s * q_rows,
                comp_raw,
                2 * comp,
                s * q_rows,
                s * groups * d.o_lora_rank,
                s * d.n_embd,
                tokens * d.n_embd,
                Table::ALL.len() * tokens * nd,
            ],
            acts: counts
                .iter()
                .flat_map(|&c| [(c, d.n_embd), (c, d.q_lora_rank)])
                .chain([(s * groups, d.group_k), (s, groups * d.o_lora_rank)])
                .collect(),
            hc_groups: tokens.div_ceil(HC_MAX_TOKENS) + 1,
            hc_k: HC_STREAMS * d.n_embd,
        };
        let need = lens.bytes();
        let (free, _) = gpu.mem_info()?;
        if tokens == 0 || need > free {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "the batch-wide attention buffers for {tokens} positions in sub-blocks of \
                     {s} take {need} B; the card budget left after the load is {free} B free"
                ),
            });
        }
        let zeroed = |i: usize| DeviceBuffer::zeroed(stream, lens.f32s[i]);
        Ok(AttnBatch {
            tokens,
            sub: s,
            rope_dims: nd,
            hc_pre: HcPreScratch::with_groups(stream, lens.hc_k, lens.hc_groups)?,
            mixes: zeroed(0)?,
            hc: zeroed(1)?,
            normed: zeroed(2)?,
            normed_acts: counts
                .iter()
                .map(|&c| act(stream, c, d.n_embd))
                .collect::<Result<_, _>>()?,
            raw_qkv: zeroed(3)?,
            q_a: zeroed(4)?,
            kv: zeroed(5)?,
            q_a_normed: zeroed(6)?,
            q_a_acts: counts
                .iter()
                .map(|&c| act(stream, c, d.q_lora_rank))
                .collect::<Result<_, _>>()?,
            raw_q: zeroed(7)?,
            q: zeroed(8)?,
            comp_raw: any_source.then(|| zeroed(9)).transpose()?,
            comp: any_source
                .then(|| PartedBuffer::zeroed(stream, [comp; 2]))
                .transpose()?,
            y: zeroed(11)?,
            heads: act(stream, s * groups, d.group_k)?,
            wo_a: zeroed(12)?,
            wo_a_act: act(stream, s, groups * d.o_lora_rank)?,
            raw_out: zeroed(13)?,
            out: zeroed(14)?,
            tables: zeroed(15)?,
            tables_host: vec![0.0; lens.f32s[15]],
            entries: 0,
            marks: Vec::new(),
            marked: 0,
            timed: false,
        })
    }

    /// The intermediate buffers the last [`AttnChain::enqueue_batch_layer`]
    /// left: HC_PRE's result and the output per token of the batch, the rest
    /// its last sub-block's (the last chunk's where a chunk writes them).
    #[must_use]
    pub fn batch_taps<'a>(&'a self, b: &'a AttnBatch) -> AttnTaps<'a> {
        let mut taps = self.taps();
        taps.hc = &b.hc;
        taps.normed = &b.normed;
        taps.q_a = &b.q_a;
        taps.q_a_normed = &b.q_a_normed;
        taps.q = &b.q;
        taps.kv = &b.kv;
        taps.y = &b.y;
        taps.wo_a = &b.wo_a;
        taps.out = &b.out;
        if let (Some(t), Some(comp)) = (taps.source.as_mut(), b.comp.as_ref()) {
            t.kv = comp.part(COMP_KV);
            t.score = comp.part(COMP_SCORE);
        }
        taps
    }

    /// Layer `layer`'s attention sub-layer over a prompt batch's chunks with
    /// its projections batch-wide (the module doc): the latent rows of the
    /// chunks `io.run ..` before `io.full` as
    /// [`AttnChain::enqueue_layer_part`] writes them, and from `io.full` on
    /// the block as [`AttnChain::enqueue_layer_staged`] runs it chunk by
    /// chunk — each launch here writes, per token, what the chunk's launch
    /// writes, so the caches, the streams and the fold are its bits. Chunk
    /// `k` reads row `k` of the piece's words and `caches`' chunk `k`.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_layer(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        b: &mut AttnBatch,
        io: BatchIo<'_>,
        caches: &mut dyn ChunkSource,
    ) -> Result<(), GpuError> {
        let AttnChain {
            layers,
            plans,
            dims,
            kernels,
            words,
            scratch,
            branch,
            top_k,
            ..
        } = self;
        let lp = layer
            .checked_sub(layers.start)
            .and_then(|i| plans.get(i))
            .ok_or_else(|| GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!("layer {layer} is not one of {layers:?}"),
            })?;
        let BatchIo {
            cuts,
            run,
            full,
            streams_in,
            fold_in,
            streams_out,
            fold_out,
            staging,
        } = io;
        let n = cuts.len();
        let base = cuts.first().map_or(0, |r| r.start);
        let tokens = cuts.last().map_or(0, |r| r.end - base);
        let chunked = cuts
            .iter()
            .all(|r| !r.is_empty() && r.end.div_ceil(HC_MAX_TOKENS) - r.start / HC_MAX_TOKENS == 1)
            && cuts.windows(2).all(|p| p[0].end == p[1].start);
        if run > full || full > n || n > words.bufs.len() || tokens > b.tokens || !chunked {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "chunks {cuts:?} from {run} (block from {full}) over {} rows of words and \
                     {} batch tokens: consecutive, cut at multiples of {HC_MAX_TOKENS}",
                    words.bufs.len(),
                    b.tokens
                ),
            });
        }
        let t = LayerTensors::of(w, lp, dims, &words.layout)?;
        let cx = BatchCx {
            gpu,
            w,
            k: kernels,
            d: dims,
            words,
            lp,
            t: &t,
            layer,
            fault: gpu.layer_sink(layer)?,
            cuts,
            base,
            top_k: *top_k,
        };
        let stream = gpu.stream();
        let (d, s4) = (cx.d, HC_STREAMS * cx.d.n_embd);
        let mut entries = 0u64;
        let t_full = cx.token(full);
        let nb = tokens - t_full;
        // Step 1: only HC_POST reads what HC_PRE writes.
        let fork = if full < n {
            let fork = branch.fork(stream)?;
            entries += 2;
            let x = span(WHAT_BATCH, streams_in, t_full * s4, nb * s4)?;
            let mut mixes = span_mut(WHAT_BATCH, &mut b.mixes, t_full * HC_MIX, nb * HC_MIX)?;
            let mut hc = span_mut(WHAT_BATCH, &mut b.hc, t_full * HC_MIX, nb * HC_MIX)?;
            cx.k.hc.enqueue_pre_groups(
                fork.stream(),
                &HcPreArgs {
                    params: &t.hc,
                    x: &x,
                    tokens: nb,
                    rms_eps: d.eps,
                    fault: cx.fault,
                },
                cuts[full].len(),
                &mut b.hc_pre,
                &mut mixes,
                &mut hc,
            )?;
            entries += 1;
            Some(fork)
        } else {
            None
        };
        for sb in SubBlocks::new(cuts, run, full) {
            b.mark(stream, &mut entries)?;
            sub_pre(&cx, b, &sb, fold_in, false, &mut entries)?;
            b.mark(stream, &mut entries)?;
            for k in sb.clone() {
                let o = cx.token(k) - cx.token(sb.start);
                let c = caches.chunk(k)?;
                chunk_rows(&cx, scratch, b, k, o, c, None, &mut entries)?;
            }
        }
        for sb in SubBlocks::new(cuts, full, n) {
            b.mark(stream, &mut entries)?;
            sub_pre(&cx, b, &sb, fold_in, true, &mut entries)?;
            b.mark(stream, &mut entries)?;
            for k in sb.clone() {
                let o = cx.token(k) - cx.token(sb.start);
                let c = caches.chunk(k)?;
                chunk_rows(&cx, scratch, b, k, o, c, Some(&mut *staging), &mut entries)?;
            }
            b.mark(stream, &mut entries)?;
            sub_post(&cx, b, &sb, &mut entries)?;
            b.mark(stream, &mut entries)?;
        }
        if let Some(fork) = fork {
            fork.join()?;
            entries += 2;
            let x = span(WHAT_BATCH, &b.out, t_full * d.n_embd, nb * d.n_embd)?;
            let res = span(WHAT_BATCH, streams_in, t_full * s4, nb * s4)?;
            let hc = span(WHAT_BATCH, &b.hc, t_full * HC_MIX, nb * HC_MIX)?;
            let mut so = span_mut(WHAT_BATCH, streams_out, t_full * s4, nb * s4)?;
            let mut fo = span_mut(WHAT_BATCH, fold_out, t_full * d.n_embd, nb * d.n_embd)?;
            cx.k.hc.enqueue_post(
                stream,
                &HcPostArgs {
                    x: &x,
                    res: &res,
                    hc: &hc,
                    n_embd: d.n_embd,
                    tokens: nb,
                },
                &mut so,
                &mut fo,
            )?;
            entries += 1;
        }
        b.entries += entries;
        Ok(())
    }
}

/// Sub-block `sb`'s launches before its chunks (module doc, step 2): the
/// attention norm of `fold_in` and its q8_1 form, the compressor's
/// projections, the joined projection with the latent part token-major, and
/// in the block (`query`) q_a token-major, its norm, the query projection
/// and the tail rope.
fn sub_pre(
    cx: &BatchCx<'_>,
    b: &mut AttnBatch,
    sb: &Range<usize>,
    fold_in: &DeviceBuffer<f32>,
    query: bool,
    entries: &mut u64,
) -> Result<(), GpuError> {
    let (d, t, gpu, stream) = (cx.d, cx.t, cx.gpu, cx.gpu.stream());
    let (ts, te) = (cx.token(sb.start), cx.token(sb.end));
    let n = te - ts;
    let groups = cx.groups(sb)?;
    let at = count_at(n, b.normed_acts.len().min(b.q_a_acts.len()))?;
    let x = span(WHAT_BATCH, fold_in, ts * d.n_embd, n * d.n_embd)?;
    cx.k.fused.enqueue_norm_quant(
        stream,
        &x,
        t.norm,
        d.eps,
        &mut b.normed_acts[at],
        &mut b.normed,
        cx.fault,
    )?;
    *entries += 1;
    let act = &b.normed_acts[at];
    if let Some(st) = &t.source {
        let missing = GpuError::State {
            what: WHAT_BATCH,
            missing: "the compressor's batch buffers",
        };
        let (raw, comp) = match (b.comp_raw.as_mut(), b.comp.as_mut()) {
            (Some(r), Some(c)) => (r, c),
            _ => return Err(missing),
        };
        let width = compress::WIDTH;
        // The gate's rows of a split pair go after a sub-block's kv rows.
        let gate_at = b.sub * width;
        let split_parts = &[(COMP_KV, 0, width, 0), (COMP_SCORE, gate_at, width, 0)];
        let parts: &[(usize, usize, usize, usize)] = match st.proj {
            SourceProj::Joint(j) => {
                gpu.enqueue_gemv_q3k_groups(j, act, groups, raw)?;
                *entries += 1;
                &[
                    (COMP_KV, 0, 2 * width, 0),
                    (COMP_SCORE, 0, 2 * width, width),
                ]
            }
            SourceProj::Split { kv, gate } => {
                let mut kv_rows = span_mut(WHAT_BATCH, raw, 0, n * width)?;
                gpu.enqueue_gemv_q3k_groups(kv, act, groups, &mut kv_rows)?;
                drop(kv_rows);
                let mut gate_rows = span_mut(WHAT_BATCH, raw, gate_at, n * width)?;
                gpu.enqueue_gemv_q3k_groups(gate, act, groups, &mut gate_rows)?;
                *entries += 2;
                split_parts
            }
            SourceProj::Kv(kv) => {
                gpu.enqueue_gemv_q3k_groups(kv, act, groups, raw)?;
                *entries += 1;
                &[(COMP_KV, 0, width, 0)]
            }
        };
        for &(part, off, total_rows, r0) in parts {
            let src = span(WHAT_BATCH, raw, off, total_rows * n)?;
            cx.k.dense.enqueue_groups_to_tokens(
                stream,
                &src,
                RowsPart {
                    total_rows,
                    r0,
                    rows: width,
                },
                groups,
                comp.part_mut(part),
            )?;
            *entries += 1;
        }
    }
    gpu.enqueue_gemv_q3k_groups(t.qkv, act, groups, &mut b.raw_qkv)?;
    *entries += 1;
    let total_rows = cx.qkv_rows();
    let latent = RowsPart {
        total_rows,
        r0: d.q_lora_rank,
        rows: d.head_dim,
    };
    cx.k.dense
        .enqueue_groups_to_tokens(stream, &b.raw_qkv, latent, groups, &mut b.kv)?;
    *entries += 1;
    if !query {
        return Ok(());
    }
    let q_a = RowsPart {
        total_rows,
        r0: 0,
        rows: d.q_lora_rank,
    };
    cx.k.dense
        .enqueue_groups_to_tokens(stream, &b.raw_qkv, q_a, groups, &mut b.q_a)?;
    cx.k.fused.enqueue_norm_quant(
        stream,
        &b.q_a,
        t.q_a_norm,
        d.eps,
        &mut b.q_a_acts[at],
        &mut b.q_a_normed,
        cx.fault,
    )?;
    gpu.enqueue_gemv_q3k_groups(t.query, &b.q_a_acts[at], groups, &mut b.raw_q)?;
    let heads_part = RowsPart {
        total_rows: cx.q_rows(),
        r0: 0,
        rows: d.n_head * d.head_dim,
    };
    cx.k.dense
        .enqueue_groups_to_tokens(stream, &b.raw_q, heads_part, groups, &mut b.q)?;
    let table = batch_table(&b.tables, b.tokens, cx.lp.tables.0, ts, n)?;
    cx.k.rope
        .enqueue_rope_tail(stream, &mut b.q, &table, heads(d, n))?;
    *entries += 5;
    Ok(())
}

/// The `n` tokens from batch token `at` of table `table` in `tables`, the
/// tables of a batch of `tokens` positions ([`AttnBatch::tables`]).
fn batch_table(
    tables: &DeviceBuffer<f32>,
    tokens: usize,
    table: Table,
    at: usize,
    n: usize,
) -> Result<View<'_, f32>, GpuError> {
    let nd = tables.len() / (Table::ALL.len() * tokens.max(1));
    view::<f32, f32>(tables, (table_index(table) * tokens + at) * nd, n * nd)
}

/// Chunk `k`'s launches in position order (module doc, step 3), its tokens
/// from token `o` of its sub-block's buffers: the compressor's rows and key
/// from its projections, the latent row — into `staging` in the block, into
/// the ring before it — and, in the block, the indexer, the staged
/// attention of the chunk's queries into the sub-block's output, and its
/// commit to the ring.
#[allow(
    clippy::too_many_arguments,
    reason = "the chunk's context, buffers, place, caches and staging (rust-quality R8)"
)]
fn chunk_rows(
    cx: &BatchCx<'_>,
    s: &mut Scratch,
    b: &mut AttnBatch,
    k: usize,
    o: usize,
    caches: ChunkCaches<'_>,
    staging: Option<&mut DeviceTensor<u16>>,
    entries: &mut u64,
) -> Result<(), GpuError> {
    let (d, t, lp, stream) = (cx.d, cx.t, cx.lp, cx.gpu.stream());
    let ccx = cx.chunk(k)?;
    let m = ccx.m;
    let ChunkCaches {
        ring,
        shadow,
        mut compressed,
        selection,
    } = caches;
    match (&lp.source, &t.source, &mut compressed) {
        (Some(sp), Some(st), Compressed::Source(io)) => {
            let width = compress::WIDTH;
            let comp = b.comp.as_ref().ok_or(GpuError::State {
                what: WHAT_BATCH,
                missing: "the compressor's batch buffers",
            })?;
            let src = s.source.as_mut().ok_or(GpuError::State {
                what: WHAT_BATCH,
                missing: "the compressor's scratch",
            })?;
            // The pooling launchers bound their reads by the stream
            // geometry's tokens, not the chunk's: the step words name the
            // chunk's rows alone, and the view runs to the geometry's extent.
            let gt = ccx.words.layout.streams[sp.stream].geom.tokens;
            let kv = view::<f32, f32>(comp.part(COMP_KV), o * width, gt * width)?;
            let score = view::<f32, f32>(comp.part(COMP_SCORE), o * width, gt * width)?;
            let raw_key = s.raw.as_mut().and_then(|r| r.key.as_mut());
            let keys = match st.keys {
                Some(_) => 3 + u64::from(raw_key.is_some() && src.act_pre.m() > 1),
                None => 0,
            };
            source_rows(
                &ccx,
                sp,
                st,
                SourceRows {
                    kv: &kv,
                    score: &score,
                    pre: &mut src.pre,
                    act_pre: &mut src.act_pre,
                    key: &mut src.key,
                    raw_key,
                },
                io,
            )?;
            *entries += 1 + keys;
        }
        (None, None, Compressed::None | Compressed::Read(_)) => {}
        (sp, _, passed) => {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "layer {} owns a compressor: {}; chunk {k}'s caches passed {}",
                    cx.layer,
                    sp.is_some(),
                    match passed {
                        Compressed::None => "no compressed rows",
                        Compressed::Read(_) => "rows to read",
                        Compressed::Source(_) => "a compressor's buffers",
                    }
                ),
            });
        }
    }
    let kv = view::<f32, f32>(&b.kv, o * d.head_dim, m * d.head_dim)?;
    let pos = ccx.words.view::<u32>(ccx.words.layout.pos, m)?;
    let table = ccx.words.view::<f32>(lp.forward, m * d.rope_dims)?;
    let Some(staging) = staging else {
        cx.k.rope.enqueue_kv_norm_rope_append(
            stream,
            KvAppendArgs {
                kv: &kv,
                gain: t.kv_norm,
                cs: &table,
                pos: &pos,
                eps: d.eps,
                n_dims: d.rope_dims,
                m,
                out: &mut s.kv_row,
                cache: ring,
                shadow,
            },
        )?;
        *entries += 1;
        return Ok(());
    };
    cx.k.rope.enqueue_kv_norm_rope_append(
        stream,
        KvAppendArgs {
            kv: &kv,
            gain: t.kv_norm,
            cs: &table,
            pos: &pos,
            eps: d.eps,
            n_dims: d.rope_dims,
            m,
            out: &mut s.kv_row,
            cache: &mut *staging,
            shadow,
        },
    )?;
    *entries += 1;
    let (rows, vis_at) = match (&compressed, lp.stream) {
        (Compressed::Read(rows), Some(st)) => (Some(*rows), ccx.words.layout.streams[st].vis),
        (Compressed::Source(io), Some(st)) => (Some(&*io.rows), ccx.words.layout.streams[st].vis),
        _ => (None, ccx.words.layout.window_vis),
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
                what: WHAT_BATCH,
                detail: format!(
                    "layer {} runs the indexer and owns its keys: {}; the caller passed them \
                     elsewhere",
                    cx.layer, ip.owns_keys
                ),
            })?;
            let kernels = cx.k.index.as_ref().ok_or(GpuError::State {
                what: WHAT_BATCH,
                missing: "the indexer's kernels",
            })?;
            let sel = s.select.as_mut().ok_or(GpuError::State {
                what: WHAT_BATCH,
                missing: "the indexer's scratch",
            })?;
            // The chunk is a column group of the sub-block's projections:
            // its block sits at `rows · o`, row-major over its m tokens.
            let (w_at, w_rows) = (d.q_lora_rank + d.head_dim, indexer::HEADS);
            let (q_at, q_len) = (d.n_head * d.head_dim, indexer::HEADS * indexer::HEAD_DIM);
            let q_in = view::<f32, f32>(&b.raw_q, cx.q_rows() * o + q_at * m, q_len * m)?;
            let w_in = view::<f32, f32>(&b.raw_qkv, cx.qkv_rows() * o + w_at * m, w_rows * m)?;
            enqueue_select(
                &ccx,
                lp,
                kernels,
                sel,
                &q_in,
                &w_in,
                IndexerIo {
                    stream: st,
                    keys,
                    list: &mut *list,
                    top_k: cx.top_k,
                },
            )?;
            *entries += 2;
            Some(&*list)
        }
        (ip, sel, st) => {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "layer {} attends stream {st:?} and runs the indexer: {}; the caller passed {}",
                    cx.layer,
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
    let q_w = d.n_head * d.head_dim;
    let q = view::<f32, f32>(&b.q, o * q_w, m * q_w)?;
    let mut y = span_mut(WHAT_BATCH, &mut b.y, o * q_w, m * q_w)?;
    let vis = ccx.words.view::<u32>(vis_at, 2 * m)?;
    let args = AttnArgs {
        q: &q,
        window: ring,
        compressed: rows,
        selected: list.map(|rows| SelectedRows {
            rows,
            stride: cx.top_k,
        }),
        vis: &vis,
        sinks: t.sinks,
        scale: d.scale,
        tokens: m,
        heads: d.n_head,
        part_v: &mut s.part_v,
        part_ms: &mut s.part_ms,
        y: &mut y,
        fault: cx.fault,
    };
    let base = ccx.words.view::<u32>(ccx.words.layout.pos, 1)?;
    let stage = StageIo {
        rows: staging,
        first: cx.cuts[k].start,
    };
    let staged = staging_view(&stage, m)?;
    let attended = cx.k.attn.enqueue_staged(
        stream,
        args,
        Staged {
            rows: &staged,
            base: &base,
        },
    );
    let committed = attended.and_then(|()| {
        cx.k.attn.enqueue_commit(
            stream,
            CommitArgs {
                staged: Staged {
                    rows: &staged,
                    base: &base,
                },
                tokens: m,
                ring: &mut *ring,
            },
        )
    });
    DeviceTensor::release(staged);
    committed?;
    *entries += 3;
    Ok(())
}

/// Sub-block `sb`'s launches after its chunks (module doc, step 4): the
/// inverse rope of the attention output, the heads' q8_1 form, wo_a's block
/// diagonal, its q8_1 form, wo_b, and wo_b's copy token-major into the
/// layer's output at the sub-block's tokens.
fn sub_post(
    cx: &BatchCx<'_>,
    b: &mut AttnBatch,
    sb: &Range<usize>,
    entries: &mut u64,
) -> Result<(), GpuError> {
    let (d, t, gpu, stream) = (cx.d, cx.t, cx.gpu, cx.gpu.stream());
    let (ts, te) = (cx.token(sb.start), cx.token(sb.end));
    let n = te - ts;
    let tokens = cx.groups(sb)?;
    let table = batch_table(&b.tables, b.tokens, cx.lp.tables.1, ts, n)?;
    cx.k.rope
        .enqueue_rope_tail(stream, &mut b.y, &table, heads(d, n))?;
    drop(table);
    let groups = d.n_head * d.head_dim / d.group_k;
    gpu.enqueue_quantize_q8_1_cols(&b.y, &mut b.heads, n * groups, cx.layer)?;
    cx.k.dense.enqueue_q3k_heads_groups(
        stream,
        Q3kHeadsGroupsArgs {
            w: t.out_a,
            act: &b.heads,
            groups,
            rows_per_head: d.o_lora_rank,
            tokens,
            y: &mut b.wo_a,
        },
    )?;
    gpu.enqueue_quantize_q8_1_cols(&b.wo_a, &mut b.wo_a_act, n, cx.layer)?;
    gpu.enqueue_gemv_q3k_groups(t.out_b, &b.wo_a_act, tokens, &mut b.raw_out)?;
    let mut out = span_mut(WHAT_BATCH, &mut b.out, ts * d.n_embd, n * d.n_embd)?;
    cx.k.dense.enqueue_groups_to_tokens(
        stream,
        &b.raw_out,
        RowsPart {
            total_rows: d.n_embd,
            r0: 0,
            rows: d.n_embd,
        },
        tokens,
        &mut out,
    )?;
    *entries += 6;
    Ok(())
}
