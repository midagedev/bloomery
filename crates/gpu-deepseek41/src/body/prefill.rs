//! The prompt batch: a prompt of `P` ids fed as `⌈P / T_MAX⌉` batches of
//! near-equal size ([`prefill`], [`batches`]), which leaves the model where
//! `P` decode steps over the same ids leave it, bit for bit, in everything
//! a later step reads: every layer's window ring, the compressed rows,
//! compressor states and index keys, the host tier, the last position's
//! logits, and the feature rows a reader keeps.
//!
//! Each layer runs only the positions a later reader needs — the CED
//! triangle ([`super::ced`]): the layers up to the last one that owns a
//! compressor or index keys run every position, and above it a layer runs
//! its block over the call's last positions and its latent rows over those
//! its window reaches. The shadow rows a layer skips are recorded as the
//! call's hole, and [`Body::keep_point`] grants no cut that would restore
//! one.
//!
//! The batches run in groups ([`groups`]; `BLOOMERY_PREFILL_GROUP` batches
//! a group, read once when the batch's buffers are made, a call's lone last
//! batch joining the group before it). A group runs its layers in order,
//! each layer over the group's batches in position order — the layer-batches
//! in that order — and each layer-batch over its batch's chunks — runs of at
//! most [`CHUNK`] positions, cut at multiples of [`CHUNK`] so a chunk's
//! latent rows fill one staging buffer from the row of its first position on
//! — from the first chunk the layer needs:
//!
//! 1. before the first layer, once per group: each batch's chunks planned in
//!    order (the host step plan, after the tokens before it), its tokens'
//!    rows read at once ([`PromptRows`]), each chunk's image built into the
//!    batch's place, and every image of the group copied to the card in one
//!    transfer with every token's rope tables ([`AttnBatch::stage_tables`]);
//!    then per batch the attention piece's gather of each chunk's words into
//!    its own row, the embedding broadcast;
//! 2. per layer-batch: the engram step where the layer carries a site, chunk
//!    by chunk; the attention sub-layer ([`AttnChain::enqueue_batch_layer`]):
//!    its projections once over sub-blocks of chunks, and chunk by chunk in
//!    position order only what carries state from a position to the next — a
//!    chunk of the block's latent rows staged and committed to the ring
//!    before the next chunk attends, a chunk before the block's first its
//!    latent rows alone; the MoE sub-layer over the block's chunks in its
//!    batch phases ([`crate::chain::ffn::FfnBatch`]) — the route of every
//!    chunk and its handoffs to the host, the card's shadow work of every
//!    chunk while one union call serves the layer's host experts for every
//!    token of the block, the sums back and one join; where the next layer
//!    is tapped and the call hands features over, the tap of the tokens whose
//!    rows are kept, one launch. In a group of two batches or more, the next
//!    layer-batch's steps up to its route are enqueued right after this one's
//!    shadow, before the host serves this one — from a layer's last batch on
//!    to the next layer's first too, whose input that batch's join wrote a
//!    layer-batch earlier — so the card routes the next layer-batch while the
//!    host serves this one; in a group of one they follow this one's join,
//!    which they read;
//! 3. after the last layer, only in the group that holds the prompt's last
//!    position: the head, for that position alone.
//!
//! With an expert tier, a layer whose tier holds experts is tiered: its
//! route also writes each slot's tier place and hands them to the host with
//! the rest; the host's serve enqueues the tier's service of the block on the
//! tier card before the union ([`bloomery_gpu::hybrid::Hybrid::serve_port`]);
//! the shadow leaves out the card sum, which the post runs once the host has
//! seen the tier's rows land ([`FfnPiece::enqueue_batch_acc_tier`]) — each
//! token's slots in slot order from either card — before the upload and the
//! join. The tier's fault word is read beside the card's at a group's end and
//! when a group fails, the tier's stream drained first.
//!
//! Under host streaming (`BLOOMERY_HOSTSTREAM=on`, a residency machine, a
//! call of at least [`STREAM_MIN_P`] positions) a group moves each layer's
//! residency pool toward the experts its first batch with a block routes
//! most, at that layer-batch: the chunks' shadow part is enqueued, the host
//! waits for the route's copies and counts the batch's ids, the machine's
//! pick sends the pool's coldest to the host and copies the hottest host
//! experts over them (`SwapMachine::call_pick`, [`STREAM_FLOOR`] a batch of
//! the group the least count admitted); the block's places are taken again
//! under the moved map, its kept card experts run, then — the engine stream
//! waiting for the pick's copies — the admitted experts run over the
//! streamed places, merged back, and the card sum reads every card slot in
//! slot order; only then is the next layer-batch's route enqueued, since it
//! writes the norm, routing and places every batch of the group shares at
//! the same offsets (the host still serves this layer-batch while the card
//! routes the next). The group's later
//! batches route under the moved map. After a layer's last batch of a group
//! the layer's reader event is recorded, which the next group's pick copies
//! wait for. The call's placement stays for the decode after it; a call that
//! fails returns each layer to the set it started with. The bits are then the
//! band's, not the decode steps'.
//!
//! Every launch writes, per token, what its one-token launch writes, and the
//! ops that carry state from a position to the next (the ring, the
//! compressor's pooling, each token's visible counts) run in position order,
//! so the batch is the steps' numbers, not an approximation of them. What a
//! layer-batch leaves for a later layer of the same batch — the streams and
//! folds, the lists, the HC_PRE results, the images and tables — is kept per
//! batch of the group; what it consumes before the next layer-batch's
//! launches write it is one buffer, since the stream runs them in order; the
//! host copies — the host tier's batch port's — come in two sets, since the
//! host reads them outside that order. The batch's buffers ([`Batch`]) and
//! the port's sets are made at load by [`prepare_prefill`], never inside a
//! call: a call on a body without them is refused by name. The decode step's
//! buffers and launches do not change.
//!
//! A call's plan — its batches, chunks, groups, the triangle's needs and each
//! layer-batch's starts and sub-blocks — is [`CallPlan`], read off the same
//! functions the enqueue runs by, so a binary prints the plan a call runs
//! before (or without) running it. Each site counts the queue entries it
//! enqueues as it enqueues them ([`Tally`]); the call's counts per batch and
//! layer-batch are [`Body::prefill_counts`], their sums [`PrefillStats`]'.
//!
//! A call that fails is taken back ([`GpuModel::rollback`]) to where it
//! found the model, unless a fault poisoned it: the fault is the model's
//! until a reset. The fault word is read at a group's end and when a group
//! fails, before anything else can fail, so a fault ends the call there,
//! named by the lowest (layer, site) any of the group's batches raised,
//! whatever error the group met after it.

use std::ops::Range;
use std::time::Instant;

use cuda_core::{CudaEvent, sys};
use gguf::quant::GgmlType;
use model::moe::UNION_MAX_COLS;

use bloomery_gpu::COL_GROUP;
use bloomery_gpu::host::swap::{CallCfg, CallPick, CallReport};
use bloomery_gpu::hybrid::BatchKey;
use bloomery_gpu::weights::DevWeight;

use super::ced::{Ced, Mode};
use super::*;
use crate::chain::attn::{AttnBatch, BatchIo, ChunkCaches, ChunkSource};
use crate::chain::ffn::{BatchLayer, BlockIo, FfnBatch, JoinIo, STAGE_TIER};
use crate::chain::glue::{GlueBatch, PromptRows};
use crate::chain::nanos;
use crate::hc::{HC_MAX_TOKENS, HC_MIX};
use crate::span::{span, span_mut};
use bloomery_gpu::Fault;
use bloomery_gpu::fault::read_cards;

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = UNION_MAX_COLS;

/// Positions one chunk runs at most: the m-column kernels' and HC_PRE's.
pub const CHUNK: usize = HC_MAX_TOKENS;
// A chunk is a column group of the batch-wide projections.
const _: () = assert!(CHUNK == COL_GROUP);

/// Chunks a batch cuts into at most: an unaligned first position adds one.
const CHUNKS_MAX: usize = T_MAX / CHUNK + 1;

/// The shortest prompt a call streams host experts for: the shortest the
/// flow model has a call plan of (`tools/flow/plans`), where it still gains
/// (prose, plan (a), +9 % at 128 positions [derived]); below it no plan
/// prices the streamed pass, so the call does not stream.
pub const STREAM_MIN_P: usize = 128;

/// The least count an expert's first batch with a block routes to it for
/// the pick to admit it, per batch the group holds (the batches the admitted
/// expert then serves from the card): the flow model's best threshold at
/// groups of one and two over P 128..16384, prose and code, plan (a) and bp,
/// which no uniform (lcg) routing's expert clears [derived].
pub const STREAM_FLOOR: u32 = 32;

const WHAT: &str = "deepseek41 prefill";

/// How a prompt is fed (`BLOOMERY_PREFILL`, [`super::BodyLevers::prefill`]):
/// in batches ([`prefill`]) or one decode step per id — the same-binary timing
/// arm, which is the decode step and not a second implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Batch,
    Steps,
}

impl PrefillMode {
    /// The mode [`PrefillMode::name`] names; `None` for any other word.
    #[must_use]
    pub fn from_name(name: &str) -> Option<PrefillMode> {
        [PrefillMode::Batch, PrefillMode::Steps]
            .into_iter()
            .find(|m| m.name() == name)
    }

    /// The name a `load` line prints, and `BLOOMERY_PREFILL` takes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PrefillMode::Batch => "batch",
            PrefillMode::Steps => "steps",
        }
    }
}

/// Make the batch's buffers: the one maker, a load-time call — a prompt
/// call ([`prefill`]) on a body without them is refused by name, so no call
/// loads a module or allocates. A load with an expert tier and the batch feed
/// makes them itself ([`super::Body::open_placed_tiered`]), before a draft
/// can attach its feature tap ([`attach_features`]); a call after the attach
/// gives the held buffers the tap's rows, which a call that hands features
/// over needs (refused by name without them). Once the buffers hold what the
/// body's tap needs, a call does nothing.
pub fn prepare_prefill(m: &mut Deepseek41Model) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.make_batch_once(gpu)
}

/// Feed `ids` from the model's position on as [`batches`] and return the
/// greedy token after the last of them; the model stands `ids.len()`
/// positions on. See the module comment.
pub fn prefill(m: &mut Deepseek41Model, ids: &[u32]) -> Result<u32, GpuError> {
    feed(m, ids, None, &mut |_, _| Ok(()))
}

/// How many batches a call of `n` positions runs: `⌈n / T_MAX⌉`.
#[must_use]
pub fn batch_count(n: usize) -> usize {
    n.div_ceil(T_MAX)
}

/// The batches a call of `n` positions from `first` runs: [`batch_count`]
/// of them, the first `n mod k` one position longer than the rest. Each
/// layer reads every host expert its batch's tokens route to once, so a
/// short last batch would pay that read for few tokens.
#[must_use]
pub fn batches(first: usize, n: usize) -> Vec<Range<usize>> {
    let k = batch_count(n);
    let mut out = Vec::with_capacity(k);
    let mut p = first;
    for j in 0..k {
        let len = n / k + usize::from(j < n % k);
        out.push(p..p + len);
        p += len;
    }
    out
}

/// A prompt call's plan as its enqueue runs it: the batches ([`batches`]),
/// the chunks each cuts into ([`chunks`]), the triangle's needs
/// ([`super::ced`]), and through its methods the groups under a group lever
/// ([`groups`]), the chunks each layer-batch starts at ([`lb_starts`]) and the
/// sub-blocks the batch-wide projections run over ([`AttnBatch::sub_blocks`])
/// — the functions a call's enqueue runs by, so a printed plan is the call's.
/// The plain call's: a call that hands its features over widens the tapped
/// layers' blocks ([`FeatureRows`]).
#[derive(Clone, Debug)]
pub struct CallPlan {
    /// `BLOOMERY_PREFILL_GROUP`: the batches a group holds.
    pub group: usize,
    /// The window ring's rows, which the triangle's latent starts reach back.
    pub ring: usize,
    pub ced: CedState,
    pub batches: Vec<Range<usize>>,
    /// Per batch, its chunks.
    pub cuts: Vec<Vec<Range<usize>>>,
    pub need: Need,
}

impl CallPlan {
    fn new(ced: &Ced, ring: usize, group: usize, first: usize, n: usize) -> CallPlan {
        let batches = batches(first, n);
        let starts: Vec<usize> = batches.iter().map(|r| r.start).collect();
        let need = ced.need(first, first + n, &starts, None);
        let cuts = batches.iter().map(|r| chunks(r.start, r.len())).collect();
        CallPlan {
            group,
            ring,
            ced: ced.state(),
            batches,
            cuts,
            need,
        }
    }

    /// The call's groups under a lever of `g`, as ranges of its batches.
    #[must_use]
    pub fn groups(&self, g: usize) -> Vec<Range<usize>> {
        groups(self.batches.len(), g)
    }

    /// The call's groups under every lever a body takes, `1 ..= GROUP_MAX`.
    #[must_use]
    pub fn every_group(&self) -> Vec<(usize, Vec<Range<usize>>)> {
        (1..=GROUP_MAX).map(|g| (g, self.groups(g))).collect()
    }

    /// Layer index `i` over batch `b`: the chunk its latent part starts at
    /// and the chunk its block starts at, the batch's chunk count for none.
    /// Panics past the call's batches.
    #[must_use]
    pub fn starts(&self, b: usize, i: usize) -> (usize, usize) {
        lb_starts(&self.need, i, &self.cuts[b])
    }

    /// Batch `b`'s chunks `k .. end` as the projections' sub-blocks. Panics
    /// past the call's batches.
    #[must_use]
    pub fn sub_blocks(&self, b: usize, k: usize, end: usize) -> Vec<Range<usize>> {
        AttnBatch::sub_blocks(&self.cuts[b], k, end)
    }
}

impl BodyLevers {
    /// The plan of a prompt call of `n` positions from `first`, on the body
    /// these levers load from `hp` with caches of `ctx_max` positions — its
    /// rings `ctx_max.min(hp.window)` rows, as the body makes them — with no
    /// card: what a call would run, before a load.
    #[must_use]
    pub fn call_plan(&self, hp: &Hparams, ctx_max: usize, first: usize, n: usize) -> CallPlan {
        let ring = ctx_max.min(hp.window);
        let ced = Ced::new(&hp.layers, ring, self.ced);
        CallPlan::new(&ced, ring, self.group, first, n)
    }
}

/// The chunks of `cuts` layer index `i` runs under `need`: its latent part
/// from the first chunk it runs at all, its block from the first it runs
/// whole — `cuts.len()` for none. A call's enqueue and its plan both read it.
fn lb_starts(need: &Need, i: usize, cuts: &[Range<usize>]) -> (usize, usize) {
    let from = |runs: fn(Mode) -> bool| {
        cuts.iter()
            .position(|r| runs(need.mode(i, r.start)))
            .unwrap_or(cuts.len())
    };
    (from(|m| m != Mode::None), from(|m| m == Mode::Full))
}

/// The queue entries one layer-batch of a prompt call enqueued — launches,
/// copies, event records, stream waits — counted where they were enqueued
/// ([`Body::prefill_counts`]), and its times, which [`PrefillStats`]' are the
/// sums of.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LbCount {
    /// The batch's index in its call, and the model layer.
    pub batch: usize,
    pub layer: usize,
    /// Whether the layer ran a block of the batch: a layer-batch
    /// [`PrefillStats::layer_batches`] counts, whose shadow the card ran and
    /// whose serve the host ran.
    pub block: bool,
    /// Its route with its upload, join and tap; its shadow.
    pub route: u64,
    pub shadow: u64,
    /// Its serve's host time: the union call, and the wait on its route's
    /// copies; 0 without a block.
    pub union_ns: u64,
    pub wait_ns: u64,
    /// With card timing on ([`Body::set_prefill_card_timing`]), its event
    /// pairs: its first launch to its route's copies (its attention alone
    /// without a block), and its shadow where it has a block.
    pub card_out_ms: Option<f64>,
    pub card_in_ms: Option<f64>,
}

impl LbCount {
    /// The count with its times left out.
    #[must_use]
    pub fn untimed(&self) -> LbCount {
        LbCount {
            batch: self.batch,
            layer: self.layer,
            block: self.block,
            route: self.route,
            shadow: self.shadow,
            ..LbCount::default()
        }
    }
}

/// The queue entries one batch's first steps enqueued: its chunks' words
/// gathered and its embedding broadcast.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontCount {
    pub batch: usize,
    pub entries: u64,
}

/// The last prompt call's counts, per batch and per layer-batch, in the order
/// its groups ran them ([`Body::prefill_counts`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PromptCounts {
    pub front: Vec<FrontCount>,
    pub lbs: Vec<LbCount>,
}

/// How far a split's time may sit from the sum of its layer-batches' before
/// [`PromptCounts::check_split`] refuses it: float sums in two orders.
pub const SPLIT_SUM_MS: f64 = 0.1;

impl PromptCounts {
    /// The counts with every layer-batch's times left out: what two runs of
    /// one call share.
    #[must_use]
    pub fn untimed(&self) -> PromptCounts {
        PromptCounts {
            front: self.front.clone(),
            lbs: self.lbs.iter().map(LbCount::untimed).collect(),
        }
    }

    /// Hold `s` to these counts when it was timed on the card: `s` covers
    /// this call alone — its batches and layer-batches — and its union, wait
    /// and card times are each the sum of the layer-batches' within
    /// [`SPLIT_SUM_MS`], so a reader of the per-layer-batch records reads the
    /// split's times. Refused by name otherwise; an untimed `s` is not held.
    pub fn check_split(&self, s: &PrefillStats) -> Result<(), GpuError> {
        const WHAT: &str = "PromptCounts::check_split";
        if !s.card_timed {
            return Ok(());
        }
        let blocks = self.lbs.iter().filter(|c| c.block).count();
        if s.batches != self.front.len() as u64 || s.layer_batches != blocks as u64 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the split covers {} batches and {} layer-batches, the counts the last \
                     call's {} and {blocks}: take the stats once per call",
                    s.batches,
                    s.layer_batches,
                    self.front.len()
                ),
            });
        }
        fn ms(ns: u64) -> f64 {
            ns as f64 / 1e6
        }
        let sum = |f: fn(&LbCount) -> f64| self.lbs.iter().map(f).sum::<f64>();
        let sums = [
            ("union_ms", ms(s.union_ns), sum(|c| ms(c.union_ns))),
            ("wait_ms", ms(s.wait_ns), sum(|c| ms(c.wait_ns))),
            (
                "card_out_ms",
                s.card_out_ms,
                sum(|c| c.card_out_ms.unwrap_or(0.0)),
            ),
            (
                "card_in_ms",
                s.card_in_ms,
                sum(|c| c.card_in_ms.unwrap_or(0.0)),
            ),
        ];
        for (name, split, lbs) in sums {
            if (split - lbs).abs() > SPLIT_SUM_MS || !(split - lbs).is_finite() {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "the split's {name} is {split:.3}, its {} layer-batches' sum {lbs:.3}: \
                         more than {SPLIT_SUM_MS} ms apart",
                        self.lbs.len()
                    ),
                });
            }
        }
        let untimed = self
            .lbs
            .iter()
            .find(|c| c.card_out_ms.is_none() || c.card_in_ms.is_some() != c.block);
        if let Some(c) = untimed {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {} of batch {} (block {}) was not timed as the split was: card_out \
                     {:?}, card_in {:?}",
                    c.layer, c.batch, c.block, c.card_out_ms, c.card_in_ms
                ),
            });
        }
        Ok(())
    }
}

/// Which sub-layer a [`BatchSeam`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchSeamKind {
    Engram,
    Attn,
    Ffn,
}

/// A point of a batch, shown to the observer of [`prefill_observed`] once
/// the launches before it are enqueued: the streams the sub-layer `kind` of
/// layer `layer` left for the tokens it ran, `tokens` of them from the
/// batch's token `at`, position `first` — `4 · n_embd` a token, the batch's
/// token `t` at `t · 4 · n_embd` — and the fold the next sub-layer reads
/// where there is one.
pub struct BatchSeam<'a> {
    pub kind: BatchSeamKind,
    pub layer: usize,
    pub first: u32,
    pub at: usize,
    pub tokens: usize,
    pub streams: &'a DeviceBuffer<f32>,
    pub fold: Option<&'a DeviceBuffer<f32>>,
    /// After an attention sub-layer: the piece's buffers as its last chunk
    /// left them.
    pub attn: Option<AttnTaps<'a>>,
}

/// [`prefill_observed`]'s observer.
pub type BatchObserver<'a> = dyn FnMut(&Gpu, BatchSeam<'_>) -> Result<(), GpuError> + 'a;

/// [`prefill`], showing `observe` each [`BatchSeam`] of every batch as its
/// launches are enqueued: a gate reads the streams there after it
/// synchronizes, and names the first seam where a batch leaves the steps.
pub fn prefill_observed(
    m: &mut Deepseek41Model,
    ids: &[u32],
    observe: &mut BatchObserver<'_>,
) -> Result<u32, GpuError> {
    feed(m, ids, None, observe)
}

/// Where [`prefill_with`] hands the feature rows: the first position they
/// hold, then one row of the tap's width per position, in order.
pub type FeatureSink<'a> = &'a mut dyn FnMut(u32, &[f32]) -> Result<(), GpuError>;

/// The feature rows a reader of a prompt call keeps: those of the call's
/// last `window` positions (all of them in a shorter call) — a draft's
/// `attention.sliding_window`, whose window-only layers hold no older row —
/// handed to `sink` in position order, one call per batch that holds any.
pub struct FeatureRows<'a> {
    pub window: usize,
    pub sink: FeatureSink<'a>,
}

/// [`prefill`], handing `features` the rows it keeps once each batch has
/// run; the tapped layers run their blocks over those positions. Refused
/// with `features` and no tap.
pub fn prefill_with(
    m: &mut Deepseek41Model,
    ids: &[u32],
    features: Option<FeatureRows<'_>>,
) -> Result<u32, GpuError> {
    feed(m, ids, features, &mut |_, _| Ok(()))
}

fn feed(
    m: &mut Deepseek41Model,
    ids: &[u32],
    features: Option<FeatureRows<'_>>,
    observe: &mut BatchObserver<'_>,
) -> Result<u32, GpuError> {
    if ids.is_empty() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: "a prompt of no ids".to_string(),
        });
    }
    let first = m.pos() as usize;
    let end = first + ids.len();
    let runs = batches(first, ids.len());
    let starts: Vec<usize> = runs.iter().map(|r| r.start).collect();
    let (window, mut sink) = match features {
        Some(f) => (Some(f.window), Some(f.sink)),
        None => (None, None),
    };
    let group = {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        let group = body.batch_mut()?.group;
        body.begin_call(first, end, &starts, window)?;
        body.stream_begin(gpu, ids.len())?;
        group
    };
    let mut token = None;
    for g in groups(runs.len(), group) {
        let rs = &runs[g.clone()];
        let (Some(b), Some(e)) = (rs.first().map(|r| r.start), rs.last().map(|r| r.end)) else {
            continue;
        };
        let ran = m
            .run_rows(e - b, WHAT, |gpu, w, body, head, _| {
                body.enqueue_group(
                    gpu,
                    w,
                    head,
                    GroupRun {
                        ids: &ids[b - first..e - first],
                        runs: rs,
                        batch: g.start,
                        last: e == end,
                    },
                    observe,
                )
            })
            .and_then(|t| {
                if let Some(f) = sink.as_mut() {
                    let (gpu, _, body) = m.body_parts(WHAT)?;
                    for (set, r) in rs.iter().enumerate() {
                        if let Some((at, rows)) = body.batch_features(gpu, set, r.clone())? {
                            f(at, rows)?;
                        }
                    }
                }
                Ok(t)
            });
        match ran {
            Ok(t) => token = t,
            Err(e) => {
                let e = stream_failed(m, e);
                return Err(take_back(m, first, e));
            }
        }
    }
    let done = token.ok_or(GpuError::State {
        what: WHAT,
        missing: "the head of the group that holds the prompt's last position",
    });
    let ended = m
        .body_parts(WHAT)
        .and_then(|(gpu, _, body)| body.stream_end(gpu, done.is_ok()));
    match (done, ended) {
        (Ok(t), Ok(())) => Ok(t),
        (Ok(_), Err(e)) => Err(take_back(m, first, e)),
        (Err(e), _) => Err(take_back(m, first, e)),
    }
}

/// A call that failed with `e` under host streaming: the machine's call
/// ended, each layer back at the set the call started with. A failure to end
/// it is named beside `e`; a fault stays `e`.
fn stream_failed(m: &mut Deepseek41Model, e: GpuError) -> GpuError {
    let ended = m
        .body_parts(WHAT)
        .and_then(|(gpu, _, body)| body.stream_end(gpu, false));
    match ended {
        Ok(()) => e,
        Err(_) if matches!(e, GpuError::Fault { .. }) => e,
        Err(s) => GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a prompt call failed ({e}), and ending its host streaming failed too ({s})"
            ),
        },
    }
}

/// A call that failed with `e`: its positions taken back, so the model
/// stands where the call found it at `first` — or, when the compressor
/// state no longer holds `first`'s group, at the cut [`Body::keep_point`]
/// grants below it, which the error then names. A fault poisons the model
/// and is returned as it came: nothing runs on it until a reset.
fn take_back(m: &mut Deepseek41Model, first: usize, e: GpuError) -> GpuError {
    if matches!(e, GpuError::Fault { .. }) || m.poisoned().is_some() {
        return e;
    }
    let kept = match m.body(WHAT) {
        Ok(body) => body.keep_point(first),
        Err(b) => return b,
    };
    let Ok(to) = u32::try_from(kept) else {
        return e;
    };
    match m.rollback(to) {
        Ok(()) if kept == first => e,
        Ok(()) => GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a prompt call from position {first} failed ({e}); its positions are taken back \
                 and the model stands at {kept}, the cut the caches grant"
            ),
        },
        Err(r) => GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a prompt call from position {first} failed ({e}), and taking it back failed \
                 too ({r})"
            ),
        },
    }
}

/// `group`, `BLOOMERY_PREFILL_GROUP` — the batches a group holds; 1 runs each
/// batch alone, every layer of it before the next batch's first — refused by
/// name, with its value, unless it is from 1 to [`GROUP_MAX`].
pub(super) fn check_group(group: usize) -> Result<(), GpuError> {
    if (1..=GROUP_MAX).contains(&group) {
        Ok(())
    } else {
        Err(GpuError::Shape {
            what: "BLOOMERY_PREFILL_GROUP",
            detail: format!("{group} batches, where a group holds 1 to {GROUP_MAX}"),
        })
    }
}

/// The largest `BLOOMERY_PREFILL_GROUP`: the lever registry's, the most its
/// row's kind takes.
const GROUP_MAX: usize = bloomery_levers::PREFILL_GROUP_MAX as usize;
const _: () = assert!(GROUP_MAX as u64 == bloomery_levers::PREFILL_GROUP_MAX);

/// Batches a group holds at most under a lever of `g`: `g`, and one more
/// from 2 on — a call's lone last batch joins the group before it
/// ([`groups`]).
fn group_sets(g: usize) -> usize {
    if g >= 2 { g + 1 } else { 1 }
}

/// The groups of a call of `k` batches under a lever of `g`: runs of `g`
/// consecutive batches, where a lone last batch joins the run before it —
/// a group of one runs no route under another batch's union. A call of one
/// batch is one group of one.
fn groups(k: usize, g: usize) -> Vec<Range<usize>> {
    let g = g.max(1);
    let mut out: Vec<Range<usize>> = (0..k).step_by(g).map(|s| s..(s + g).min(k)).collect();
    if g >= 2
        && out.len() >= 2
        && out.last().is_some_and(|r| r.len() == 1)
        && let Some(tail) = out.pop()
        && let Some(prev) = out.last_mut()
    {
        prev.end = tail.end;
    }
    out
}

/// A batch's own buffers in its group: per token its streams and folds,
/// ping and pong; per chunk, per indexer layer, its list of [`CHUNK`]
/// tokens; with a feature tap, per token its row.
struct BatchSet {
    hc: [DeviceBuffer<f32>; 2],
    folds: [DeviceBuffer<f32>; 2],
    lists: Vec<Vec<DeviceBuffer<u32>>>,
    taps: Option<DeviceBuffer<f32>>,
}

impl BatchSet {
    /// A batch's feature rows for a tap of `width` values a row: [`T_MAX`]
    /// rows; none without a tap (`width` 0). Batch sets made before the tap
    /// get theirs from [`Body::complete_batch_taps`].
    fn tap_rows(stream: &CudaStream, width: usize) -> Result<Option<DeviceBuffer<f32>>, GpuError> {
        (width > 0)
            .then(|| DeviceBuffer::<f32>::zeroed(stream, T_MAX * width))
            .transpose()
            .map_err(GpuError::from)
    }

    fn device_bytes(&self) -> usize {
        self.hc.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
            + self
                .folds
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .lists
                .iter()
                .flatten()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.taps.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// The batch's buffers: see the module comment. Made once, for groups of up
/// to `sets.len()` batches.
pub(super) struct Batch {
    /// `BLOOMERY_PREFILL_GROUP` as the buffers were made for it.
    group: usize,
    /// The chunk images' host builder, the chunks' plans, every batch's
    /// chunk images laid end to end on the host — [`CHUNKS_MAX`] a batch —
    /// and their card copy.
    image: StepImage,
    plans: Vec<StepPlan>,
    images: Vec<u32>,
    params: DeviceBuffer<u32>,
    /// The attention piece over images of [`CHUNK`] tokens, a row of words
    /// per chunk of every batch of a group, and its batch-wide buffers.
    attn: AttnChain,
    proj: AttnBatch,
    glue: GlueBatch,
    ffn: FfnBatch,
    rows: PromptRows,
    /// Per batch of a group, its own buffers.
    sets: Vec<BatchSet>,
    /// A chunk's latent rows before its commit, row `p % CHUNK` for position
    /// `p`.
    staging: DeviceTensor<u16>,
    /// A batch's feature rows on the host.
    taps_host: Vec<f32>,
    /// The host and card time of the batches since the last
    /// [`Body::take_prefill_stats`].
    stats: PrefillStats,
    /// Whether each layer-batch's card work is timed by events
    /// ([`CARD_MARKS`] a layer of each batch of a group, recorded only while
    /// this is on).
    card_timing: bool,
    card_marks: Vec<CudaEvent>,
    /// Per batch of a group, per layer: whether the last group ran its block
    /// there, so its shadow pair was recorded.
    card_served: Vec<bool>,
    /// The last call's queue entries, per batch and layer-batch.
    counts: PromptCounts,
    /// The call's host streaming ([`Body::stream_begin`]).
    stream: StreamCall,
}

/// A call's host streaming: whether this call streams, the group it is at,
/// each pick so far with its group, and the last call's end.
#[derive(Default)]
struct StreamCall {
    on: bool,
    group: usize,
    picks: Vec<(usize, CallPick)>,
    end: Option<CallReport>,
    /// A pick's counts, one an expert of the layer.
    counts: Vec<u32>,
}

/// Events a layer-batch records while [`Body::set_prefill_card_timing`] is
/// on: before its first launch, where its route's copies to the host
/// complete (after its attention where it has no block), and before and
/// after its shadow — which, in a group, the stream reaches after the
/// previous layer-batch's upload and join.
const CARD_MARKS: usize = 4;

/// Where a batch's time went, summed over the batches since the last
/// [`Body::take_prefill_stats`]. Host times are wall clock on the calling
/// thread; the card times are event pairs on the engine stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrefillStats {
    /// `BLOOMERY_PREFILL_GROUP` the batches ran under.
    pub group: usize,
    /// Batches run, and layer-batches that served the host tier; of them
    /// those of a group's first batch.
    pub batches: u64,
    pub layer_batches: u64,
    pub first_layer_batches: u64,
    /// The groups' host half before the launches ([`Body::plan_group`]):
    /// the plans, the rows, the images and their copy, which waits for the
    /// stream.
    pub prologue_ns: u64,
    /// The launches' enqueue with every serve in it.
    pub chain_ns: u64,
    /// Of `chain_ns`: the union calls, the waits on the route's copies and
    /// the activations' copy into the union's view; of the waits, those of a
    /// group's first batch — the layer-batches whose route the previous
    /// layer's last batch enqueued.
    pub union_ns: u64,
    pub wait_ns: u64,
    pub wait_first_ns: u64,
    pub copy_ns: u64,
    /// Card time, with [`Body::set_prefill_card_timing`] on: from each
    /// layer-batch's first launch to its route's copies (its attention alone
    /// where it has no block), and its shadow, which runs under the union;
    /// and the attention's batch-wide projection phases, summed.
    pub card_timed: bool,
    pub card_out_ms: f64,
    pub card_in_ms: f64,
    pub card_proj_ms: f64,
    /// Entries the batches put in the launch queue — launches, copies, event
    /// records, stream waits, counted where each is enqueued:
    /// those of each batch's first steps and each layer-batch's route (the
    /// engram step, the attention, the route and its copies) with its
    /// upload, join and tap, and those of each shadow.
    pub entries_route: u64,
    pub entries_shadow: u64,
    /// Of the host slots the batch services listed, those the service's
    /// exclusion set skipped (`HybridStats::batch_excluded_slots`).
    pub excluded_slots: u64,
}

impl PrefillStats {
    /// Of `chain_ns`, the time the thread spent enqueueing: the rest once
    /// the union calls, the waits and the copies are taken out.
    #[must_use]
    pub fn enqueue_ns(&self) -> u64 {
        self.chain_ns
            .saturating_sub(self.union_ns + self.wait_ns + self.copy_ns)
    }

    /// A group's sums `o` added to these; `group` stays. Every field is
    /// named, so a new sum does not compile until it is added here.
    fn add(&mut self, o: &PrefillStats) {
        let PrefillStats {
            group: _,
            batches,
            layer_batches,
            first_layer_batches,
            prologue_ns,
            chain_ns,
            union_ns,
            wait_ns,
            wait_first_ns,
            copy_ns,
            card_timed,
            card_out_ms,
            card_in_ms,
            card_proj_ms,
            entries_route,
            entries_shadow,
            excluded_slots,
        } = *o;
        self.batches += batches;
        self.layer_batches += layer_batches;
        self.first_layer_batches += first_layer_batches;
        self.prologue_ns += prologue_ns;
        self.chain_ns += chain_ns;
        self.union_ns += union_ns;
        self.wait_ns += wait_ns;
        self.wait_first_ns += wait_first_ns;
        self.copy_ns += copy_ns;
        self.card_timed |= card_timed;
        self.card_out_ms += card_out_ms;
        self.card_in_ms += card_in_ms;
        self.card_proj_ms += card_proj_ms;
        self.entries_route += entries_route;
        self.entries_shadow += entries_shadow;
        self.excluded_slots += excluded_slots;
    }
}

impl Batch {
    /// Refused by name ([`no_tap_rows`]) unless every batch set holds the
    /// feature rows of a tap of `width`: checked before a call that hands
    /// features over enqueues anything.
    fn check_taps(&self, width: usize, tiered: bool) -> Result<(), GpuError> {
        let sets = self.sets.len();
        match self
            .sets
            .iter()
            .position(|s| s.taps.as_ref().map(DeviceBuffer::len) != Some(T_MAX * width))
        {
            Some(k) => Err(no_tap_rows(k, sets, width, tiered)),
            None => Ok(()),
        }
    }

    pub(super) fn set_top_k(&mut self, gpu: &Gpu, top_k: usize) -> Result<(), GpuError> {
        self.attn.set_top_k(gpu, top_k)
    }

    fn device_bytes(&self) -> usize {
        self.params.num_bytes()
            + self.attn.device_bytes()
            + self.proj.device_bytes()
            + self.glue.device_bytes()
            + self.ffn.device_bytes()
            + self.sets.iter().map(BatchSet::device_bytes).sum::<usize>()
            + self.staging.buf().num_bytes()
    }

    /// Of [`Batch::device_bytes`], what each batch past a group's first holds
    /// for itself: its [`BatchSet`], and its share of the buffers laid out a
    /// batch at a time — the images' card copy, the attention's rows of
    /// words, the rope tables and the HC_PRE results.
    fn group_bytes(&self) -> usize {
        let sets = self.sets.len();
        let shared = self.params.num_bytes()
            + self.attn.words_bytes()
            + self.proj.tables_bytes()
            + self.ffn.hc_bytes();
        let own = self.sets.first().map_or(0, BatchSet::device_bytes);
        (sets - 1) * (own + shared / sets)
    }
}

/// One group of a prompt: its ids, its batches' positions, the index of its
/// first batch in the call, and whether it holds the prompt's last position
/// (the head's).
struct GroupRun<'a> {
    ids: &'a [u32],
    runs: &'a [Range<usize>],
    batch: usize,
    last: bool,
}

/// A batch of a group as its layers run: its index in the call, its set of
/// buffers, its chunks, its first position and tokens, and which half of its
/// ping-pong pairs its next sub-layer reads.
struct Member {
    batch: usize,
    set: usize,
    cuts: Vec<Range<usize>>,
    b: usize,
    u: usize,
    cur: Cursor,
}

impl Member {
    /// The batch's token where the chunk `k` on starts, `u` past the last.
    fn token(&self, k: usize) -> usize {
        self.cuts.get(k).map_or(self.u, |r| r.start - self.b)
    }

    /// The host tier's batch key of model layer `layer`'s block of the
    /// batch, from its token `at` on.
    fn key(&self, layer: usize, at: usize) -> BatchKey {
        BatchKey {
            layer,
            set: self.set,
            at,
            u: self.u,
        }
    }
}

/// What a layer-batch's route leaves for its shadow, serve and join: the
/// block's first chunk and token (the batch's end where it has none), and
/// the layer's resolved tensors where it has one.
struct Routed<'w> {
    full: usize,
    at: usize,
    block: Option<BatchLayer<'w>>,
}

/// A layer's caches with each chunk's list ([`BatchSet::lists`]), as
/// [`AttnChain::enqueue_batch_layer`] asks for them chunk by chunk.
struct LayerCaches<'a> {
    kv: &'a mut [LayerKv],
    shadows: &'a mut Shadows,
    lists: &'a mut [Vec<DeviceBuffer<u32>>],
    i: usize,
    step: LayerStep,
}

impl ChunkSource for LayerCaches<'_> {
    fn chunk(&mut self, k: usize) -> Result<ChunkCaches<'_>, GpuError> {
        let lists = self.lists.get_mut(k).ok_or(GpuError::State {
            what: WHAT,
            missing: "the chunk's lists",
        })?;
        let LayerIo {
            ring,
            shadow,
            compressed,
            selection,
        } = layer_io(self.kv, self.shadows, lists, self.i, &self.step)?;
        Ok(ChunkCaches {
            ring,
            shadow,
            compressed,
            selection,
        })
    }
}

/// Where a group's launch sites count the queue entries they enqueue —
/// launches, copies, event records, stream waits — as they enqueue them: per
/// batch its first steps, per layer-batch its route with its upload, join and
/// tap, and its shadow. The attention piece counts its own launches into its
/// layer-batch's route ([`BatchIo::entries`]); every other site adds what its
/// call enqueued right after the call returns, the count its callee's
/// enqueue makes written beside it. The group's sums and the call's
/// [`PromptCounts`] are read from here once the group has run.
struct Tally {
    front: Vec<FrontCount>,
    /// Layer by layer, each over the group's batches: the group's order.
    lbs: Vec<LbCount>,
    batches: usize,
}

impl Tally {
    /// A group of `batches` batches, the first the call's batch `batch`,
    /// over the model layers `layers`.
    fn new(batch: usize, batches: usize, layers: Range<usize>) -> Tally {
        Tally {
            front: (0..batches)
                .map(|j| FrontCount {
                    batch: batch + j,
                    entries: 0,
                })
                .collect(),
            lbs: layers
                .flat_map(|layer| {
                    (0..batches).map(move |j| LbCount {
                        batch: batch + j,
                        layer,
                        ..LbCount::default()
                    })
                })
                .collect(),
            batches,
        }
    }

    /// The count of layer index `i` over the group's batch `set`.
    fn lb(&mut self, i: usize, set: usize) -> Result<&mut LbCount, GpuError> {
        let n = self.lbs.len();
        (set < self.batches)
            .then(|| self.lbs.get_mut(i * self.batches + set))
            .flatten()
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer index {i} of batch {set} of a group of {} batches ({n} layer-batches)",
                    self.batches
                ),
            })
    }

    /// `n` entries of batch `set`'s first steps.
    fn front(&mut self, set: usize, n: u64) -> Result<(), GpuError> {
        let batches = self.batches;
        let f = self.front.get_mut(set).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("batch {set} of a group of {batches} batches"),
        })?;
        f.entries += n;
        Ok(())
    }

    /// `n` entries of layer index `i`'s route over batch `set`.
    fn route(&mut self, i: usize, set: usize, n: u64) -> Result<(), GpuError> {
        self.lb(i, set)?.route += n;
        Ok(())
    }

    /// `n` entries of layer index `i`'s shadow over batch `set`.
    fn shadow(&mut self, i: usize, set: usize, n: u64) -> Result<(), GpuError> {
        self.lb(i, set)?.shadow += n;
        Ok(())
    }

    /// The group's entries as the stat line sums them: every batch's first
    /// steps and layer-batch's route, and every shadow.
    fn sums(&self) -> (u64, u64) {
        let front: u64 = self.front.iter().map(|f| f.entries).sum();
        let route: u64 = self.lbs.iter().map(|c| c.route).sum();
        let shadow: u64 = self.lbs.iter().map(|c| c.shadow).sum();
        (front + route, shadow)
    }
}

/// Whether the resident weight `name` is a Q3_K or Q4_K row stream, which a
/// launch reads in q8_1: the q8_1 form is a launch of its own.
fn reads_q8(w: &Weights, name: &str) -> bool {
    matches!(
        w.get(name),
        Some(DevWeight::KQuant {
            ty: GgmlType::Q3_K | GgmlType::Q4_K,
            ..
        })
    )
}

/// The entries `FfnPiece::enqueue_batch_shadow` enqueues for the block
/// `chunks` of layer `l`, with card experts or not: per chunk HC_PRE and the
/// norm, the norm's q8_1 codes and scales copied into the block's planes when
/// there are card experts, the shared expert's gate·up (one launch for a
/// chunk when gate and up are both Q3_K, else one a token), its down's q8_1
/// form when the down reads one, the down and past one token its copy
/// token-major; then over the block the buckets, the tile table, the gather
/// into run order, the gate·up, its q8_1 form and the down when there are
/// card experts, and the card sum — or, on a `tiered` layer, whose card sum
/// the post runs, the three copies of the block's routing it holds for it.
fn shadow_entries(w: &Weights, l: usize, chunks: &[Range<usize>], card: bool, tiered: bool) -> u64 {
    let q3k = |name: &str| {
        matches!(
            w.get(name),
            Some(DevWeight::KQuant {
                ty: GgmlType::Q3_K,
                ..
            })
        )
    };
    let gate_up_q3k = q3k(&names::ffn_gate_shexp(l)) && q3k(&names::ffn_up_shexp(l));
    let down_q8 = reads_q8(w, &names::ffn_down_shexp(l));
    let per_chunk: u64 = chunks
        .iter()
        .map(|r| {
            let m = r.len() as u64;
            let routed = 2 * u64::from(card);
            let gate_up = if gate_up_q3k { 1 } else { m };
            2 + routed + gate_up + u64::from(down_q8) + 1 + u64::from(m > 1)
        })
        .sum();
    per_chunk + if card { 6 } else { 0 } + if tiered { 3 } else { 1 }
}

/// The positions `b .. b + u` cut into chunks: at every multiple of
/// [`CHUNK`], so a chunk never passes the staging's last row.
fn chunks(b: usize, u: usize) -> Vec<Range<usize>> {
    let mut out = Vec::with_capacity(u / CHUNK + 2);
    let (mut p, end) = (b, b + u);
    while p < end {
        let next = ((p / CHUNK + 1) * CHUNK).min(end);
        out.push(p..next);
        p = next;
    }
    out
}

impl Body {
    /// The batch's buffers, once [`prepare_prefill`] has made them; refused
    /// by name before, since making them inside a call loads modules and
    /// allocates — driver calls that may wait for a card's streams, where a
    /// stalled tier would hang the call instead of failing it by name.
    fn batch_mut(&mut self) -> Result<&mut Batch, GpuError> {
        self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the prompt batch's buffers, made at load (body::prepare_prefill)",
        })
    }

    /// Make the batch's buffers when none are held: the attention piece over
    /// the batch layout, the glue's and the MoE sub-layer's scratch, each
    /// batch's streams, folds, lists and images for a group of the most
    /// batches `BLOOMERY_PREFILL_GROUP` gives, the staging, the host tier's
    /// batch sets for as many tokens as the MoE scratch takes, the host
    /// union's scratch. Held buffers made before the feature tap get its
    /// rows ([`Body::complete_batch_taps`]). Load-time only
    /// ([`prepare_prefill`]).
    fn make_batch_once(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.batch.is_none() {
            let b = self.make_batch(gpu)?;
            self.hybrid.prepare_batch(gpu.context(), b.ffn.cap())?;
            self.hybrid.host_mut().prepare_union()?;
            self.batch = Some(Box::new(b));
        }
        self.complete_batch_taps(gpu)
    }

    /// Give every batch set of the held buffers that has no feature rows the
    /// tap's ([`BatchSet::tap_rows`]), and the host rows their width: a
    /// tiered load makes the buffers before a draft can attach its tap
    /// ([`super::Body::open_placed_tiered`], then [`attach_features`]). A set
    /// whose rows are of another width is refused by name; nothing to do
    /// without a tap or without buffers.
    fn complete_batch_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let width = self.tap.as_ref().map_or(0, FeatureTap::width);
        let Some(batch) = self.batch.as_deref_mut() else {
            return Ok(());
        };
        if width == 0 {
            return Ok(());
        }
        let (group, sets) = (batch.group, batch.sets.len());
        for (k, set) in batch.sets.iter_mut().enumerate() {
            match set.taps.as_ref().map(DeviceBuffer::len) {
                Some(len) if len == T_MAX * width => {}
                Some(len) => {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "batch {k} of a group of {sets} holds {len} feature values, the tap's \
                             {T_MAX} rows of {width}"
                        ),
                    });
                }
                None => {
                    set.taps =
                        BatchSet::tap_rows(gpu.stream(), width).map_err(|e| GpuError::Shape {
                            what: WHAT,
                            detail: format!(
                                "the feature rows of batch {k} of a group of {sets} \
                                 (BLOOMERY_PREFILL_GROUP={group}) do not fit on the card: {e}"
                            ),
                        })?;
                }
            }
        }
        batch.taps_host.resize(T_MAX * width, 0.0);
        Ok(())
    }

    fn make_batch(&self, gpu: &Gpu) -> Result<Batch, GpuError> {
        let hp = &self.hp;
        let stream = gpu.stream();
        let group = self.levers.group;
        let sets = group_sets(group);
        let row_bytes = engram_row_bytes(&self.file, hp)?;
        let dims = ImageDims::of(hp, &self.planner, CHUNK, row_bytes);
        let (window, yarn) = rope_specs(hp)?;
        let image = StepImage::new(ImageLayout::new(dims)?, &window, &yarn)?;
        let words = image.layout().words();
        let mut attn = AttnChain::with_rows(
            gpu,
            hp,
            self.layers.clone(),
            image.layout(),
            &self.planner,
            CHUNKS_MAX * sets,
        )?;
        if attn.top_k() != self.attn.top_k() {
            attn.set_top_k(gpu, self.attn.top_k())?;
        }
        let writers = self.lists.first().map_or(0, Vec::len);
        let list_len = attn.list_len();
        let n = hp.n_embd;
        let tap_width = self.tap.as_ref().map_or(0, FeatureTap::width);
        let params = DeviceBuffer::zeroed(stream, sets * CHUNKS_MAX * words)?;
        let glue = self.glue.batch(gpu, image.layout())?;
        let mut ffn = FfnBatch::new(gpu, n, hp.experts.ff, [T_MAX, sets], self.tier.is_some())?;
        if self.levers.hoststream {
            ffn.enable_stream(stream)?;
        }
        let set = |k: usize| -> Result<BatchSet, GpuError> {
            let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
            let made = || -> Result<BatchSet, GpuError> {
                Ok(BatchSet {
                    hc: [z(T_MAX * HC_STREAMS * n)?, z(T_MAX * HC_STREAMS * n)?],
                    folds: [z(T_MAX * n)?, z(T_MAX * n)?],
                    lists: (0..CHUNKS_MAX)
                        .map(|_| {
                            (0..writers)
                                .map(|_| DeviceBuffer::zeroed(stream, CHUNK * list_len))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    taps: BatchSet::tap_rows(stream, tap_width)?,
                })
            };
            made().map_err(|e| GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the buffers of batch {k} of a group of {sets} (BLOOMERY_PREFILL_GROUP={group}) \
                     do not fit on the card: {e}"
                ),
            })
        };
        let sets_made = (0..sets).map(set).collect::<Result<Vec<_>, _>>()?;
        let staging = DeviceTensor::zeroed(stream, CHUNK, hp.head_dim)?;
        let card_marks = (0..CARD_MARKS * self.layers.len() * sets)
            .map(|_| {
                gpu.context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Last: its refusal names the card's free bytes with every other
        // buffer of the batch already taken.
        let proj = attn.batch(gpu, T_MAX, sets)?;
        Ok(Batch {
            group,
            plans: vec![StepPlan::default(); CHUNKS_MAX],
            images: vec![0; sets * CHUNKS_MAX * words],
            params,
            glue,
            ffn,
            rows: self.rows.prompt_rows(T_MAX),
            sets: sets_made,
            staging,
            taps_host: vec![0.0; T_MAX * tap_width],
            stats: PrefillStats::default(),
            card_timing: false,
            card_marks,
            card_served: vec![false; self.layers.len() * sets],
            counts: PromptCounts::default(),
            stream: StreamCall::default(),
            image,
            attn,
            proj,
        })
    }

    /// Per layer index of the body, the experts the stage card's tier
    /// ([`STAGE_TIER`]) holds of the layer, as the slot map says now; all 0
    /// without a tier. With a tier, a layer the slot map has no row for is
    /// refused by name. Read once per group.
    fn tier_counts(&self) -> Result<Vec<usize>, GpuError> {
        let map = self.slot_map();
        let tier = !self.hybrid.tiers().is_empty();
        self.layers
            .clone()
            .map(|l| {
                if tier {
                    map.on_tier_of(STAGE_TIER, l)
                } else {
                    Ok(0)
                }
            })
            .collect()
    }

    /// The prompt batches' time since the last call, and zero again; the
    /// default before the first batch.
    pub fn take_prefill_stats(&mut self) -> PrefillStats {
        self.batch
            .as_deref_mut()
            .map(|b| PrefillStats {
                group: b.group,
                ..std::mem::take(&mut b.stats)
            })
            .unwrap_or_default()
    }

    /// Time each layer-batch's card work with events from the next group on
    /// ([`PrefillStats`]); off, no event is recorded. Refused before the
    /// batch's buffers are made ([`prepare_prefill`]).
    pub fn set_prefill_card_timing(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let batch = self.batch_mut()?;
        batch.card_timing = on;
        let layer_batches = layers * batch.sets.len();
        batch.proj.set_card_timing(gpu, on, layer_batches)
    }

    /// Whether the next prompt calls stream host experts: the load's
    /// `BLOOMERY_HOSTSTREAM` until set here, so one load can run both arms.
    /// Refused by name: before the batch's buffers are made
    /// ([`prepare_prefill`]), on a batch made without the streamed places
    /// (the lever off at the load), and while a call streams.
    pub fn set_hoststream(&mut self, on: bool) -> Result<(), GpuError> {
        let batch = self.batch_mut()?;
        if batch.stream.on {
            return Err(GpuError::State {
                what: "Body::set_hoststream",
                missing: "no call streaming",
            });
        }
        if on && !batch.ffn.streams() {
            return Err(GpuError::State {
                what: "Body::set_hoststream",
                missing: "a batch made with the streamed places (BLOOMERY_HOSTSTREAM=on at the load)",
            });
        }
        self.levers.hoststream = on;
        Ok(())
    }

    /// Device bytes of the batch's buffers; 0 before they are made
    /// ([`prepare_prefill`]).
    #[must_use]
    pub fn batch_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.device_bytes())
    }

    /// The part of [`Body::batch_bytes`] the batch-wide attention
    /// projections hold ([`crate::chain::attn::AttnBatch`]); 0 before they are
    /// made.
    #[must_use]
    pub fn batch_proj_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.proj.device_bytes())
    }

    /// `BLOOMERY_PREFILL_GROUP` as the batch's buffers were made for it, and
    /// the part of [`Body::batch_bytes`] the batches past a group's first
    /// hold for themselves; `None` before the first batch.
    #[must_use]
    pub fn prefill_group(&self) -> Option<(usize, usize)> {
        self.batch.as_ref().map(|b| (b.group, b.group_bytes()))
    }

    /// `BLOOMERY_PREFILL_GROUP` as the body was loaded with it: the batches a
    /// group of the batch's buffers holds, made or not.
    #[must_use]
    pub fn prefill_group_lever(&self) -> usize {
        self.levers.group
    }

    /// The queue entries the last prompt call enqueued, per batch and per
    /// layer-batch, from the groups that ran whole; `None` before the first
    /// batch.
    #[must_use]
    pub fn prefill_counts(&self) -> Option<&PromptCounts> {
        self.batch.as_ref().map(|b| &b.counts)
    }

    /// A prompt call of positions `first .. end`, fed as batches from
    /// `starts`, whose reader keeps the features of its last `window`
    /// positions when `window` is given: its needs ([`super::ced`]), kept for
    /// the batches and [`Body::prefill_need`]. Its hole is recorded when its
    /// first group has planned ([`Body::plan_group`]). Refused past the
    /// positions the body computes the reference at ([`Body::check_defined`]),
    /// with a window and no tap, and on a card that does not run every layer.
    fn begin_call(
        &mut self,
        first: usize,
        end: usize,
        starts: &[usize],
        window: Option<usize>,
    ) -> Result<(), GpuError> {
        self.check_defined(WHAT, end)?;
        if self.layers.start != 0 || self.layers.end != self.hp.n_layer {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a card that runs every layer",
            });
        }
        let after: Option<Vec<bool>> = match (window, self.tap.as_ref()) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(GpuError::State {
                    what: WHAT,
                    missing: "a feature tap (attach_features)",
                });
            }
            (Some(_), Some(tap)) => {
                if let Some(b) = self.batch.as_deref() {
                    b.check_taps(tap.width(), self.tier.is_some())?;
                }
                Some(tap.after.iter().map(Option::is_some).collect())
            }
        };
        let taps = after.as_deref().zip(window);
        self.need = Some(self.ced.need(first, end, starts, taps));
        if let Some(b) = self.batch.as_deref_mut() {
            b.counts.front.clear();
            b.counts.lbs.clear();
        }
        Ok(())
    }

    /// A call of `n` positions under host streaming: with the lever on and
    /// `n` at least [`STREAM_MIN_P`], the residency machine opens its call
    /// ([`Hybrid::call_begin`]) at [`STREAM_FLOOR`] a batch of the lever's
    /// group, and the call streams; else it does not. Refused by name: a call
    /// streaming already, and the lever on with no machine (the load refuses
    /// it beside `BLOOMERY_RESIDENCY=off`).
    fn stream_begin(&mut self, gpu: &Gpu, n: usize) -> Result<(), GpuError> {
        let group = self.levers.group;
        let on = self.levers.hoststream && n >= STREAM_MIN_P;
        let batch = self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        if batch.stream.on {
            return Err(GpuError::State {
                what: WHAT,
                missing: "no call streaming (stream_end)",
            });
        }
        batch.stream.picks.clear();
        batch.stream.group = 0;
        if !on {
            return Ok(());
        }
        let floor = STREAM_FLOOR.saturating_mul(u32::try_from(group).unwrap_or(u32::MAX));
        // Each group sets its own floor before its picks
        // (`enqueue_group_chain`); this one stands for none.
        if !self.hybrid.call_begin(gpu.stream(), CallCfg { floor })? {
            return Err(GpuError::State {
                what: "BLOOMERY_HOSTSTREAM=on",
                missing: "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>)",
            });
        }
        batch.stream.on = true;
        Ok(())
    }

    /// The streaming call's end ([`Hybrid::call_end`]): its placement `kept`
    /// for the decode after it, else each layer back at the set the call
    /// started with. Nothing for a call that does not stream.
    fn stream_end(&mut self, gpu: &Gpu, kept: bool) -> Result<(), GpuError> {
        let Some(batch) = self.batch.as_deref_mut() else {
            return Ok(());
        };
        if !std::mem::take(&mut batch.stream.on) {
            return Ok(());
        }
        batch.stream.end = self.hybrid.call_end(gpu.stream(), kept)?;
        Ok(())
    }

    /// The last streaming call's picks, each with its group, and its end;
    /// empty and `None` after a call that did not stream. Taken: a second
    /// read is empty.
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        match self.batch.as_deref_mut() {
            Some(b) => (std::mem::take(&mut b.stream.picks), b.stream.end.take()),
            None => (Vec::new(), None),
        }
    }

    /// The feature rows the group's batch `set` of positions `run` kept —
    /// those of its positions from the call's first kept one on, `None` when
    /// it holds none — and the first position they hold, copied to the host
    /// in one transfer (a blocking read on the engine stream). Refused
    /// without a tap.
    fn batch_features(
        &mut self,
        gpu: &Gpu,
        set: usize,
        run: Range<usize>,
    ) -> Result<Option<(u32, &[f32])>, GpuError> {
        let width = self.tap.as_ref().map_or(0, FeatureTap::width);
        let from = self
            .need
            .as_ref()
            .map_or(run.end, |n| n.features)
            .max(run.start);
        let tiered = self.tier.is_some();
        let batch = self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "a batch that ran",
        })?;
        let sets = batch.sets.len();
        let dev = batch
            .sets
            .get(set)
            .and_then(|s| s.taps.as_ref())
            .ok_or_else(|| no_tap_rows(set, sets, width, tiered))?;
        if run.is_empty() || run.len() > T_MAX {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("feature rows of a batch {run:?} of at most {T_MAX}"),
            });
        }
        if from >= run.end {
            return Ok(None);
        }
        let (at, u) = (from - run.start, run.len());
        let rows = span(WHAT, dev, at * width, (u - at) * width)?;
        let host = &mut batch.taps_host[at * width..u * width];
        rows.copy_to_host(gpu.stream(), host)?;
        let pos = u32::try_from(from).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("position {from} passes u32"),
        })?;
        Ok(Some((pos, host)))
    }

    /// One group: the prompt's ids `ids` as the batches `runs`, the head only
    /// when `last`. Returns whether it enqueued the head. See the module
    /// comment. A group reads the fault word back at its end (a blocking
    /// read) and returns the fault as the named error: the lowest (layer,
    /// site) any of its batches raised. A group that fails after its first
    /// launch waits for the stream before it returns, so the call is taken
    /// back with no copy in flight, and reads the fault word too
    /// ([`fault_or`]).
    fn enqueue_group(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        run: GroupRun<'_>,
        observe: &mut BatchObserver<'_>,
    ) -> Result<bool, GpuError> {
        let GroupRun {
            ids,
            runs,
            batch,
            last,
        } = run;
        self.admit(WHAT, Entry::Group)?;
        let stream = gpu.stream();
        if capturing(stream)? {
            return Err(GpuError::State {
                what: WHAT,
                missing: "an eager stream: a batch is served while it is enqueued",
            });
        }
        self.arrive()?;
        self.rows.finish()?;
        let sets = self.batch_mut()?.sets.len();
        let b = runs.first().map_or(0, |r| r.start);
        let end = runs.last().map_or(b, |r| r.end);
        let joined = runs.windows(2).all(|p| p[0].end == p[1].start);
        let sized = runs.iter().all(|r| (1..=T_MAX).contains(&r.len()));
        if self.history.len() != b
            || runs.is_empty()
            || runs.len() > sets
            || !joined
            || !sized
            || ids.len() != end - b
            || end > self.positions()
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a group of batches {runs:?} (each 1..={T_MAX} positions, consecutive, at \
                     most {sets} of them) of {} ids after {} tokens, in caches of {} positions",
                    ids.len(),
                    self.history.len(),
                    self.positions()
                ),
            });
        }
        if self
            .need
            .as_ref()
            .is_none_or(|n| b < n.first || end > n.end)
        {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a call's needs that cover the group (begin_call)",
            });
        }
        if self.restore {
            for run in self.holds.stale_runs(b) {
                for (i, layer) in self.kv.iter_mut().enumerate() {
                    self.shadows
                        .restore(i, &mut layer.ring, stream, run.clone())?;
                }
                run.for_each(|q| self.holds.ring_wrote(q));
            }
            self.restore = false;
        }
        let mut members: Vec<Member> = runs
            .iter()
            .enumerate()
            .map(|(set, r)| Member {
                batch: batch + set,
                set,
                cuts: chunks(r.start, r.len()),
                b: r.start,
                u: r.len(),
                cur: Cursor::default(),
            })
            .collect();
        let t0 = Instant::now();
        self.plan_group(stream, ids, &members)?;
        let prologue = nanos(t0.elapsed());
        for p in b..end {
            self.holds.wrote(p);
        }
        if let Some(tap) = self.tap.as_mut() {
            tap.pos = [None; PAIR_ROWS];
        }
        let excluded0 = self.hybrid.stats().batch_excluded_slots;
        let t1 = Instant::now();
        let chained = self.enqueue_group_chain(gpu, w, head, &mut members, last, observe);
        let chain = nanos(t1.elapsed());
        let (mut sums, tally) = match chained {
            Ok(done) => done,
            Err(e) => {
                let tier = self.hybrid.tier_fault();
                return Err(fault_or(gpu, tier, b, e));
            }
        };
        // Before anything else can fail: a fault the group raised on either
        // card is the call's error, never the next call's; the first layer
        // wins.
        let card = gpu.fault()?;
        let tier = self.hybrid.tier_fault()?;
        if let Some(fault) = read_cards(&[card, tier]) {
            return Err(GpuError::fault(WHAT, fault));
        }
        sums.chain_ns = chain;
        sums.excluded_slots = self
            .hybrid
            .stats()
            .batch_excluded_slots
            .saturating_sub(excluded0);
        sums.prologue_ns = prologue;
        self.account_group(members.len(), sums, tally)?;
        Ok(last)
    }

    /// Add one group's sums, its `g` batches, its queue entries and, with
    /// card timing on, its layer-batches' event pairs — read first, which
    /// waits for the last of them — to the stats all at once: a group that
    /// fails adds nothing. The group's union, wait and card times are the
    /// sums of its layer-batches'.
    fn account_group(
        &mut self,
        g: usize,
        mut sums: PrefillStats,
        mut tally: Tally,
    ) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let batch = self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        sums.batches = g as u64;
        if batch.card_timing {
            sums.card_timed = true;
            sums.card_proj_ms = batch.proj.take_card_ms()?;
            let (marks, _) = batch.card_marks.as_chunks::<CARD_MARKS>();
            // The tally runs layer by layer over the group's batches, the
            // events batch by batch over the layers.
            for (x, lb) in tally.lbs.iter_mut().enumerate() {
                let at = (x % g) * layers + x / g;
                let (Some([e0, e1, e2, e3]), Some(&served)) =
                    (marks.get(at), batch.card_served.get(at))
                else {
                    return Err(GpuError::State {
                        what: WHAT,
                        missing: "a card event pair for each layer-batch of the group",
                    });
                };
                if served != lb.block {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "layer {} of batch {}: its events say block {served}, its count \
                             block {}",
                            lb.layer, lb.batch, lb.block
                        ),
                    });
                }
                lb.card_out_ms = Some(f64::from(e0.elapsed_ms(e1)?));
                lb.card_in_ms = if served {
                    Some(f64::from(e2.elapsed_ms(e3)?))
                } else {
                    None
                };
            }
        }
        for lb in &tally.lbs {
            sums.union_ns += lb.union_ns;
            sums.wait_ns += lb.wait_ns;
            sums.card_out_ms += lb.card_out_ms.unwrap_or(0.0);
            sums.card_in_ms += lb.card_in_ms.unwrap_or(0.0);
        }
        (sums.entries_route, sums.entries_shadow) = tally.sums();
        batch.stats.add(&sums);
        batch.counts.front.extend(tally.front);
        batch.counts.lbs.extend(tally.lbs);
        Ok(())
    }

    /// The group's host half: each batch's chunks planned in order, each
    /// after the tokens before it (the ids joining the history), each
    /// batch's rows read and its chunks' images built into its own place,
    /// then every batch's images copied to the card in one transfer and its
    /// rope tables in another, each of which synchronizes the stream. The
    /// call's first group records the call's hole once the history holds
    /// the group's positions: a hole always starts below the history's
    /// length, so taking the call back to its first position drops it.
    fn plan_group(
        &mut self,
        stream: &CudaStream,
        ids: &[u32],
        members: &[Member],
    ) -> Result<(), GpuError> {
        let Body {
            batch,
            planner,
            history,
            file,
            holes,
            need,
            ..
        } = self;
        let batch = batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        let first = members.first().map_or(0, |m| m.b);
        let words = batch.image.layout().words();
        let mut used = 0;
        for m in members {
            let n = m.cuts.len();
            for (plan, r) in batch.plans.iter_mut().zip(&m.cuts) {
                let toks = &ids[r.start - first..r.end - first];
                planner
                    .plan_into(toks, r.start as u32, history, plan)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                history.extend_from_slice(toks);
            }
            batch.rows.fill(file, &batch.plans[..n])?;
            for (k, r) in m.cuts.iter().enumerate() {
                let at = r.start - m.b..r.end - m.b;
                batch.image.build(
                    &batch.plans[k],
                    batch.rows.embd(at.clone())?,
                    batch.rows.engram(at.clone())?,
                )?;
                let row = m.set * CHUNKS_MAX + k;
                batch.images[row * words..(row + 1) * words].copy_from_slice(batch.image.words());
                batch
                    .proj
                    .stage_tables(batch.image.layout(), batch.image.words(), m.set, at)?;
            }
            used = m.set * CHUNKS_MAX + n;
        }
        if let Some(need) = need.as_ref()
            && need.first == first
        {
            let hole = need.hole();
            if !hole.is_empty() {
                holes.push(hole);
            }
        }
        let mut dst = span_mut(WHAT, &mut batch.params, 0, used * words)?;
        dst.copy_from_host(stream, &batch.images[..used * words])?;
        batch.proj.upload_tables(stream, members.len())
    }

    /// The group's launches (the module comment's steps 1 to 3 on the card),
    /// each layer-batch over the chunks the call's needs give it, the host
    /// tier served layer-batch by layer-batch while they are enqueued: in a
    /// group of two or more, a layer-batch's route is enqueued right after
    /// the shadow of the one before it, ahead of that one's serve; in a group
    /// of one, after its join, which the route reads.
    fn enqueue_group_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        members: &mut [Member],
        last: bool,
        observe: &mut BatchObserver<'_>,
    ) -> Result<(PrefillStats, Tally), GpuError> {
        let on_tier = self.tier_counts()?;
        let Body {
            layers,
            kv,
            shadows,
            steps,
            slots,
            hybrid,
            ffn,
            glue,
            tap,
            batch,
            hp,
            need,
            tier,
            picked,
            ..
        } = self;
        let batch = batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        let need = need.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the call's needs",
        })?;
        if batch.stream.on {
            let n = u32::try_from(members.len()).unwrap_or(u32::MAX);
            hybrid.call_floor(STREAM_FLOOR.saturating_mul(n))?;
        }
        let first = members.first().map_or(0, |m| m.batch);
        let mut cx = GroupCx {
            gpu,
            w,
            layers: layers.clone(),
            kv: &mut kv[..],
            shadows,
            steps: &steps[..],
            slots,
            hybrid,
            ffn,
            glue: &mut *glue,
            tap: tap.as_ref(),
            batch: &mut *batch,
            need,
            tier: tier.as_ref(),
            on_tier,
            n: hp.n_embd,
            sums: PrefillStats::default(),
            tally: Tally::new(first, members.len(), layers.clone()),
        };
        for m in members.iter_mut() {
            cx.front(m)?;
        }
        let g = members.len();
        let items: Vec<(usize, usize)> = (0..layers.len())
            .flat_map(|i| (0..g).map(move |j| (i, j)))
            .collect();
        // The route runs ahead of the host's serve only where the next
        // layer-batch is another batch's: in a group of one it reads this
        // one's join.
        let ahead = g >= 2;
        let streaming = cx.batch.stream.on;
        // Per layer index, whether this group's pick has run there: the
        // body's load-sized bitset, zeroed for this group.
        *picked = 0;
        let mut next = match items.first() {
            Some(&(i, j)) => Some(cx.route(&mut members[j], i, observe)?),
            None => None,
        };
        for (x, &(i, j)) in items.iter().enumerate() {
            let now = next.take().ok_or(GpuError::State {
                what: WHAT,
                missing: "the route of the layer-batch the group serves",
            })?;
            let l = layers.start + i;
            let card = streaming && CardStacks::of(w, l)?.is_some();
            if card && (*picked >> i) & 1 == 0 && now.block.is_some() {
                *picked |= 1 << i;
                cx.shadow_pick(&members[j], i, &now)?;
                cx.shadow_stream(&members[j], i, &now)?;
            } else {
                cx.shadow(&members[j], i, &now)?;
            }
            // The next route writes the block buffers (its norm, routing and
            // places) every batch of the group shares at the same offsets:
            // it follows this layer-batch's last reader of them, the card
            // sum (or the held routing's copy).
            if ahead && let Some(&(i2, j2)) = items.get(x + 1) {
                next = Some(cx.route(&mut members[j2], i2, observe)?);
            }
            if card && j + 1 == g {
                cx.hybrid.call_reader(l, gpu.stream())?;
            }
            cx.serve(&members[j], i, &now)?;
            cx.post(&mut members[j], i, &now, observe)?;
            if !ahead && let Some(&(i2, j2)) = items.get(x + 1) {
                next = Some(cx.route(&mut members[j2], i2, observe)?);
            }
        }
        let GroupCx { sums, tally, .. } = cx;
        if streaming {
            batch.stream.group += 1;
        }
        if last && let Some(m) = members.last() {
            let s4 = HC_STREAMS * hp.n_embd;
            let t = m.u - 1;
            let own = batch.sets.get(m.set).ok_or(GpuError::State {
                what: WHAT,
                missing: "the last batch's buffers",
            })?;
            let s = span(WHAT, &own.hc[m.cur.s], t * s4, s4)?;
            let pre = span(WHAT, batch.ffn.hc(m.set)?, t * HC_MIX, HC_MIX)?;
            glue.enqueue_head(gpu, w, &s, &pre, head)?;
        }
        Ok((sums, tally))
    }
}

/// What a group's layer-batches enqueue through: the body's parts they read
/// and write, the batch's buffers, the call's needs, and the group's sums so
/// far — the layer-batches, the serves' waits, the queue entries each site
/// enqueued — which reach the stats only once the group has run
/// ([`Body::account_group`]).
struct GroupCx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    layers: Range<usize>,
    kv: &'a mut [LayerKv],
    shadows: &'a mut Shadows,
    steps: &'a [LayerStep],
    slots: &'a DeviceTensor<u32>,
    hybrid: &'a mut Hybrid<Ds41Host>,
    ffn: &'a mut FfnPiece,
    glue: &'a mut Glue,
    tap: Option<&'a FeatureTap>,
    batch: &'a mut Batch,
    need: &'a Need,
    /// The stage card's tier piece, and per layer index the experts the
    /// expert tier holds of the layer.
    tier: Option<&'a TierPiece>,
    on_tier: Vec<usize>,
    n: usize,
    sums: PrefillStats,
    tally: Tally,
}

impl<'a> GroupCx<'a> {
    /// The experts the expert tier holds of layer index `i`
    /// ([`Body::tier_counts`], read at the group's start); a layer with any
    /// is tiered.
    fn on_tier(&self, i: usize) -> Result<usize, GpuError> {
        self.on_tier.get(i).copied().ok_or(GpuError::State {
            what: WHAT,
            missing: "the layer's tier count: the group reads one for each of the body's layers",
        })
    }

    /// Whether layer index `i` is tiered.
    fn tiered(&self, i: usize) -> Result<bool, GpuError> {
        Ok(self.on_tier(i)? > 0)
    }

    /// Event `k` of the layer-batch (layer index `i`, batch `set`).
    fn mark(&self, set: usize, i: usize, k: usize) -> Result<(), GpuError> {
        let at = CARD_MARKS * (set * self.layers.len() + i) + k;
        let e = self.batch.card_marks.get(at).ok_or(GpuError::State {
            what: WHAT,
            missing: "a card event for the layer-batch: the pool is sized for a group",
        })?;
        e.record(self.gpu.stream())?;
        Ok(())
    }

    /// A batch's first steps, before its first layer: its chunks' words
    /// gathered into its rows, the embedding broadcast into its streams and
    /// fold.
    fn front(&mut self, m: &mut Member) -> Result<(), GpuError> {
        let (gpu, n) = (self.gpu, self.n);
        let batch = &mut *self.batch;
        let words = batch.image.layout().words();
        let s4 = HC_STREAMS * n;
        let row0 = m.set * CHUNKS_MAX;
        for k in 0..m.cuts.len() {
            let p = span(WHAT, &batch.params, (row0 + k) * words, words)?;
            batch.attn.enqueue_step_of(gpu, &p, row0 + k)?;
            // The chunk's words gathered into its row: one launch.
            self.tally.front(m.set, 1)?;
        }
        let own = set_of(&mut batch.sets, m.set)?;
        for (k, r) in m.cuts.iter().enumerate() {
            let (at, len) = (r.start - m.b, r.len());
            let p = span(WHAT, &batch.params, (row0 + k) * words, words)?;
            let [h0, _] = &mut own.hc;
            let [f0, _] = &mut own.folds;
            let mut s = span_mut(WHAT, h0, at * s4, len * s4)?;
            let mut f = span_mut(WHAT, f0, at * n, len * n)?;
            self.glue
                .enqueue_batch_embed(gpu, &batch.glue, &p, len, &mut s, &mut f)?;
            // The embedding broadcast: a launch a token.
            self.tally.front(m.set, len as u64)?;
        }
        m.cur = Cursor::default();
        Ok(())
    }

    /// Layer index `i` of the batch `m` up to its route: the engram step
    /// where the layer carries a site, the attention sub-layer, and where the
    /// layer runs a block of the batch, the MoE sub-layer's route of it and
    /// its copies to the host. Shows `observe` the engram's and the
    /// attention's seams.
    fn route(
        &mut self,
        m: &mut Member,
        i: usize,
        observe: &mut BatchObserver<'_>,
    ) -> Result<Routed<'a>, GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let l = self.layers.start + i;
        let step = self.steps[i];
        let timed = self.batch.card_timing;
        let tiered = self.tiered(i)?;
        let served = m.set * self.layers.len() + i;
        if timed {
            self.mark(m.set, i, 0)?;
            self.tally.route(i, m.set, 1)?;
        }
        self.batch.card_served[served] = false;
        // The layer runs a suffix of the chunks: its latent part from
        // `run`, its block from `full`.
        let (run, full) = lb_starts(self.need, i, &m.cuts);
        if step.engram {
            self.engram(m, i, run, observe)?;
        }
        let batch = &mut *self.batch;
        let row0 = m.set * CHUNKS_MAX;
        let (u, b) = (m.u, m.b);
        let own = set_of(&mut batch.sets, m.set)?;
        {
            let count = self.tally.lb(i, m.set)?;
            let (sin, sout) = ping(&mut own.hc, m.cur.s);
            let (fin, fout) = ping(&mut own.folds, m.cur.f);
            let mut caches = LayerCaches {
                kv: &mut self.kv[..],
                shadows: &mut *self.shadows,
                lists: &mut own.lists[..],
                i,
                step,
            };
            batch.attn.enqueue_batch_layer(
                gpu,
                w,
                l,
                &mut batch.proj,
                BatchIo {
                    set: m.set,
                    cuts: &m.cuts,
                    row: row0,
                    run,
                    full,
                    streams_in: sin,
                    fold_in: fin,
                    streams_out: sout,
                    fold_out: fout,
                    staging: &mut batch.staging,
                    entries: &mut count.route,
                },
                &mut caches,
            )?;
        }
        m.cur.s ^= 1;
        m.cur.f ^= 1;
        let at = m.token(full);
        observe(
            gpu,
            BatchSeam {
                kind: BatchSeamKind::Attn,
                layer: l,
                first: (b + at) as u32,
                at,
                tokens: u - at,
                streams: &own.hc[m.cur.s],
                fold: Some(&own.folds[m.cur.f]),
                attn: Some(batch.attn.batch_taps(&batch.proj)),
            },
        )?;
        if full == m.cuts.len() {
            if timed {
                self.mark(m.set, i, 1)?;
                self.tally.route(i, m.set, 1)?;
            }
            return Ok(Routed {
                full,
                at,
                block: None,
            });
        }
        batch.card_served[served] = true;
        self.tally.lb(i, m.set)?.block = true;
        self.sums.layer_batches += 1;
        if m.set == 0 {
            self.sums.first_layer_batches += 1;
        }
        let block = BlockIo {
            set: m.set,
            chunks: &m.cuts[full..],
            base: b,
            streams: &own.hc[m.cur.s],
            fold_in: &own.folds[m.cur.f],
        };
        let bl = self.ffn.resolve_batch(w, l)?;
        self.ffn
            .enqueue_batch_route(gpu, &bl, &mut batch.ffn, &block, self.slots)?;
        // Each chunk's norm, the router's two launches over the block, the
        // places.
        self.tally
            .route(i, m.set, (m.cuts.len() - full) as u64 + 3)?;
        if tiered {
            let map = self.tier.map(TierPiece::places).ok_or(GpuError::State {
                what: WHAT,
                missing: "the stage card's tier piece of a body with a tiered layer",
            })?;
            self.ffn
                .enqueue_batch_tier_places(gpu, &bl, &mut batch.ffn, &block, map)?;
            // The slots' tier places.
            self.tally.route(i, m.set, 1)?;
        }
        batch
            .ffn
            .enqueue_download(gpu, self.hybrid, m.key(l, at), tiered)?;
        // The three copies to the host — four on a tiered layer, its tier
        // places — and the event the host waits on.
        self.tally.route(i, m.set, 4 + u64::from(tiered))?;
        if timed {
            self.mark(m.set, i, 1)?;
            self.tally.route(i, m.set, 1)?;
        }
        Ok(Routed {
            full,
            at,
            block: Some(bl),
        })
    }

    /// Layer index `i`'s engram step over the batch `m`'s chunks from chunk
    /// `run` on, chunk by chunk. Shows `observe` its seam.
    fn engram(
        &mut self,
        m: &mut Member,
        i: usize,
        run: usize,
        observe: &mut BatchObserver<'_>,
    ) -> Result<(), GpuError> {
        let (gpu, w, n) = (self.gpu, self.w, self.n);
        let l = self.layers.start + i;
        let batch = &mut *self.batch;
        let words = batch.image.layout().words();
        let s4 = HC_STREAMS * n;
        let row0 = m.set * CHUNKS_MAX;
        let (u, b) = (m.u, m.b);
        let own = set_of(&mut batch.sets, m.set)?;
        let wkv_q8 = reads_q8(w, &names::engram_wkv(l));
        for (k, r) in m.cuts.iter().enumerate().skip(run) {
            let (at, len) = (r.start - b, r.len());
            let p = span(WHAT, &batch.params, (row0 + k) * words, words)?;
            let (hin, hout) = ping(&mut own.hc, m.cur.s);
            let streams = span(WHAT, hin, at * s4, len * s4)?;
            let mut out = span_mut(WHAT, hout, at * s4, len * s4)?;
            let pre = span(WHAT, batch.ffn.hc(m.set)?, at * HC_MIX, len * HC_MIX)?;
            let mut input = span_mut(WHAT, &mut own.folds[m.cur.f], at * n, len * n)?;
            self.glue.enqueue_batch_engram(
                gpu,
                w,
                &mut batch.glue,
                l,
                &p,
                len,
                EngramStep {
                    streams: &streams,
                    pre: &pre,
                    out: &mut out,
                    input: &mut input,
                },
            )?;
            // A row launch a token, the q8_1 form when wkv reads one, the wkv
            // projection and past one token its copy token-major, the key
            // norm, the gate and the fold.
            self.tally.route(
                i,
                m.set,
                len as u64 + u64::from(wkv_q8) + 1 + u64::from(len > 1) + 3,
            )?;
        }
        m.cur.s ^= 1;
        let at = m.token(run);
        observe(
            gpu,
            BatchSeam {
                kind: BatchSeamKind::Engram,
                layer: l,
                first: (b + at) as u32,
                at,
                tokens: u - at,
                streams: &own.hc[m.cur.s],
                fold: Some(&own.folds[m.cur.f]),
                attn: None,
            },
        )
    }

    /// Layer index `i`'s shadow of the batch `m`'s block, where it has one.
    fn shadow(&mut self, m: &Member, i: usize, r: &Routed<'_>) -> Result<(), GpuError> {
        let Some(bl) = r.block.as_ref() else {
            return Ok(());
        };
        let (gpu, w) = (self.gpu, self.w);
        let l = self.layers.start + i;
        let timed = self.batch.card_timing;
        let tiered = self.tiered(i)?;
        if timed {
            self.mark(m.set, i, 2)?;
            self.tally.shadow(i, m.set, 1)?;
        }
        let batch = &mut *self.batch;
        let own = set_of(&mut batch.sets, m.set)?;
        let card = CardStacks::of(w, l)?;
        let entries = shadow_entries(w, l, &m.cuts[r.full..], card.is_some(), tiered);
        let block = BlockIo {
            set: m.set,
            chunks: &m.cuts[r.full..],
            base: m.b,
            streams: &own.hc[m.cur.s],
            fold_in: &own.folds[m.cur.f],
        };
        self.ffn
            .enqueue_batch_shadow(gpu, bl, card, &mut batch.ffn, &block, tiered)?;
        self.tally.shadow(i, m.set, entries)?;
        if timed {
            self.mark(m.set, i, 3)?;
            self.tally.shadow(i, m.set, 1)?;
        }
        Ok(())
    }

    /// Under host streaming, layer index `i`'s shadow of the batch `m`'s
    /// block, the group's first at the layer, up to its streamed pass: each
    /// chunk's part, then — the host waiting for the route's copies — the
    /// layer's pick from the block's routed ids, the block's places taken
    /// again under the moved map, and its kept card experts.
    fn shadow_pick(&mut self, m: &Member, i: usize, r: &Routed<'_>) -> Result<(), GpuError> {
        let bl = r.block.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "a block for the layer's pick",
        })?;
        let (gpu, w) = (self.gpu, self.w);
        let l = self.layers.start + i;
        let tiered = self.tiered(i)?;
        if self.batch.card_timing {
            self.mark(m.set, i, 2)?;
            self.tally.shadow(i, m.set, 1)?;
        }
        let batch = &mut *self.batch;
        let own = set_of(&mut batch.sets, m.set)?;
        let card = CardStacks::of(w, l)?;
        let block = BlockIo {
            set: m.set,
            chunks: &m.cuts[r.full..],
            base: m.b,
            streams: &own.hc[m.cur.s],
            fold_in: &own.folds[m.cur.f],
        };
        self.ffn
            .enqueue_batch_shadow_chunks(gpu, bl, card, &mut batch.ffn, &block)?;
        let n_expert = self.hybrid.slots().n_expert();
        let counts = &mut batch.stream.counts;
        counts.clear();
        counts.resize(n_expert, 0);
        for &id in self.hybrid.routed_ids(m.key(l, r.at))? {
            let c = usize::try_from(id)
                .ok()
                .and_then(|id| counts.get_mut(id))
                .ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {l}: a routed id {id} of {n_expert} experts"),
                })?;
            *c += 1;
        }
        let pick = self
            .hybrid
            .call_pick(gpu.stream(), l, &batch.stream.counts, usize::MAX)?;
        batch.stream.picks.push((batch.stream.group, pick));
        self.ffn
            .enqueue_batch_replace(gpu, bl, &mut batch.ffn, &block, self.slots)?;
        self.ffn
            .enqueue_batch_block_kept(gpu, bl, card, &mut batch.ffn, &block)?;
        // The shadow's entries and the places taken again.
        let entries = shadow_entries(w, l, &m.cuts[r.full..], card.is_some(), tiered);
        self.tally.shadow(i, m.set, entries + 1)
    }

    /// Under host streaming, the streamed pass of layer index `i`'s block of
    /// the batch `m` after [`GroupCx::shadow_pick`]: the engine stream waits
    /// for the pick's copies, then the admitted experts, the places merged
    /// and the card sum (or the held routing).
    fn shadow_stream(&mut self, m: &Member, i: usize, r: &Routed<'_>) -> Result<(), GpuError> {
        let bl = r.block.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "a block for the layer's streamed pass",
        })?;
        let (gpu, w) = (self.gpu, self.w);
        let l = self.layers.start + i;
        let tiered = self.tiered(i)?;
        let landed = self.hybrid.call_landed(l)?.ok_or(GpuError::State {
            what: WHAT,
            missing: "a residency machine's call for the streamed pass",
        })?;
        gpu.stream().wait(landed)?;
        let batch = &mut *self.batch;
        let own = set_of(&mut batch.sets, m.set)?;
        let card = CardStacks::of(w, l)?;
        let block = BlockIo {
            set: m.set,
            chunks: &m.cuts[r.full..],
            base: m.b,
            streams: &own.hc[m.cur.s],
            fold_in: &own.folds[m.cur.f],
        };
        self.ffn
            .enqueue_batch_stream(gpu, bl, card, &mut batch.ffn, &block, tiered)?;
        // The wait, the tile path's six launches over the streamed places
        // with card experts, and the merge.
        self.tally
            .shadow(i, m.set, 2 + if card.is_some() { 6 } else { 0 })?;
        if self.batch.card_timing {
            self.mark(m.set, i, 3)?;
            self.tally.shadow(i, m.set, 1)?;
        }
        Ok(())
    }

    /// Layer index `i`'s host experts for the batch `m`'s block, where it
    /// has one: the wait on its route's copies and one union call, whose
    /// times are its layer-batch's.
    fn serve(&mut self, m: &Member, i: usize, r: &Routed<'_>) -> Result<(), GpuError> {
        if r.block.is_none() {
            return Ok(());
        }
        let l = self.layers.start + i;
        let union0 = self.hybrid.stats().batch_ns;
        let times = self.batch.ffn.serve(&mut *self.hybrid, m.key(l, r.at))?;
        let union = self.hybrid.stats().batch_ns.saturating_sub(union0);
        let lb = self.tally.lb(i, m.set)?;
        lb.union_ns = union;
        lb.wait_ns = times.wait_ns;
        let st = &mut self.sums;
        st.copy_ns += times.copy_ns;
        if m.set == 0 {
            st.wait_first_ns += times.wait_ns;
        }
        Ok(())
    }

    /// Layer index `i` of the batch `m` after its serve: the host sums'
    /// upload and the join where it has a block, the streams the layer
    /// leaves, and where the next layer is tapped, the tap of the kept rows.
    /// Shows `observe` the MoE sub-layer's seam.
    fn post(
        &mut self,
        m: &mut Member,
        i: usize,
        r: &Routed<'_>,
        observe: &mut BatchObserver<'_>,
    ) -> Result<(), GpuError> {
        let (gpu, n) = (self.gpu, self.n);
        let l = self.layers.start + i;
        let step = self.steps[i];
        let s4 = HC_STREAMS * n;
        let n_tier = self.on_tier(i)?;
        let tiered = n_tier > 0;
        let batch = &mut *self.batch;
        let sets = batch.sets.len();
        let own = set_of(&mut batch.sets, m.set)?;
        if r.block.is_some() {
            let key = m.key(l, r.at);
            if tiered {
                let rows = self.hybrid.tier_rows_of(key, STAGE_TIER)?;
                self.ffn.enqueue_batch_acc_tier(
                    gpu,
                    &mut batch.ffn,
                    [l, m.set, n_tier],
                    [r.at, m.u],
                    rows,
                )?;
                // The card sum over both cards' slots, one launch.
                self.tally.route(i, m.set, 1)?;
            }
            batch.ffn.enqueue_upload(gpu, self.hybrid, key)?;
            // The host sums' copy to the card.
            self.tally.route(i, m.set, 1)?;
            let (sin, sout) = ping(&mut own.hc, m.cur.s);
            let (_, fout) = ping(&mut own.folds, m.cur.f);
            self.ffn.enqueue_batch_join(
                gpu,
                &batch.ffn,
                l,
                r.at,
                m.u,
                JoinIo {
                    set: m.set,
                    streams: sin,
                    streams_out: sout,
                    fold_out: step.folds.then_some(fout),
                },
            )?;
            // The join, one launch.
            self.tally.route(i, m.set, 1)?;
        }
        m.cur.s ^= 1;
        if step.folds {
            m.cur.f ^= 1;
        }
        observe(
            gpu,
            BatchSeam {
                kind: BatchSeamKind::Ffn,
                layer: l,
                first: (m.b + r.at) as u32,
                at: r.at,
                tokens: m.u - r.at,
                streams: &own.hc[m.cur.s],
                fold: step.folds.then_some(&own.folds[m.cur.f]),
                attn: None,
            },
        )?;
        // The tap of the kept rows: every one of them is in the block.
        let kept = self.need.features.max(m.b) - m.b;
        if kept < m.u
            && let Some(tap) = self.tap
            && let Some(slot) = tap.after.get(i).copied().flatten()
        {
            let width = tap.width();
            let dev = own
                .taps
                .as_mut()
                .ok_or_else(|| no_tap_rows(m.set, sets, width, self.tier.is_some()))?;
            let k = m.u - kept;
            let s = span(WHAT, &own.hc[m.cur.s], kept * s4, k * s4)?;
            let mut rows = span_mut(WHAT, dev, kept * width, k * width)?;
            batch
                .ffn
                .enqueue_tap_means(gpu, &s, k, width, slot * n, &mut rows)?;
            // The tap's means of the kept rows, one launch.
            self.tally.route(i, m.set, 1)?;
        }
        Ok(())
    }
}

/// The error of a group from position `b` that failed with `e` after its
/// first launch: the stream waited for, then the fault word read, merged
/// with `tier`, the expert tier's word as read after its stream drained
/// (the first layer wins) — a fault any of the group's launches raised on
/// either card is the error whatever `e` was, with `e` behind it, so it
/// poisons the model, never reaches a later call, and keeps `e`'s text (a
/// host refusal's layer, say). A fault `e` that names the same fault stays as
/// it is: it names the reader that met it first. A failed wait or read names
/// `e` beside its own error, and a fault `e` stays.
fn fault_or(gpu: &Gpu, tier: Result<Option<Fault>, GpuError>, b: usize, e: GpuError) -> GpuError {
    let read = gpu.fault().and_then(|card| Ok(read_cards(&[card, tier?])));
    match read {
        Ok(Some(fault)) if matches!(&e, GpuError::Fault { fault: seen, .. } if *seen == fault) => e,
        Ok(Some(fault)) => GpuError::Fault {
            what: WHAT,
            fault,
            behind: Some(Box::new(e)),
        },
        Ok(None) => e,
        Err(_) if matches!(e, GpuError::Fault { .. }) => e,
        Err(s) => GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a group from position {b} failed ({e}), and waiting for its launches or \
                 reading the fault word failed too ({s})"
            ),
        },
    }
}

/// The refusal of a call that hands features over while batch `set` of a
/// group of `sets` holds no feature rows of the tap's `width`, naming the
/// load: its buffers were made before the tap and never completed
/// ([`Body::complete_batch_taps`]).
fn no_tap_rows(set: usize, sets: usize, width: usize, tiered: bool) -> GpuError {
    let load = if tiered {
        "a load with an expert tier card"
    } else {
        "a load with no tier card"
    };
    GpuError::Shape {
        what: WHAT,
        detail: format!(
            "batch {set} of a group of {sets} holds no feature rows of the tap's {T_MAX} rows of \
             {width}, on {load}: the batch's buffers were made before the tap (attach_features) \
             and prepare_prefill did not run after it"
        ),
    }
}

/// The group's batch `set`'s own buffers, refused past the group.
fn set_of(sets: &mut [BatchSet], set: usize) -> Result<&mut BatchSet, GpuError> {
    let n = sets.len();
    sets.get_mut(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("batch {set} of a group of at most {n}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A timed call of two batches over three layers, the first layer
    /// without a block, and its stats as the group's fold makes them.
    fn call() -> (PromptCounts, PrefillStats) {
        let mut lbs = Vec::new();
        for layer in 0..3 {
            for batch in 0..2 {
                let block = layer > 0;
                lbs.push(LbCount {
                    batch,
                    layer,
                    block,
                    route: 10,
                    shadow: u64::from(block) * 5,
                    union_ns: u64::from(block) * 30_000_000 + batch as u64,
                    wait_ns: u64::from(block) * 1_500_000,
                    card_out_ms: Some(20.0 + layer as f64 * 0.25),
                    card_in_ms: block.then_some(12.5),
                });
            }
        }
        let front = (0..2)
            .map(|batch| FrontCount { batch, entries: 3 })
            .collect();
        let counts = PromptCounts { front, lbs };
        let mut s = PrefillStats {
            batches: 2,
            layer_batches: 4,
            card_timed: true,
            ..PrefillStats::default()
        };
        for c in &counts.lbs {
            s.union_ns += c.union_ns;
            s.wait_ns += c.wait_ns;
            s.card_out_ms += c.card_out_ms.unwrap_or(0.0);
            s.card_in_ms += c.card_in_ms.unwrap_or(0.0);
        }
        (counts, s)
    }

    fn refused(counts: &PromptCounts, s: &PrefillStats) -> String {
        match counts.check_split(s) {
            Err(GpuError::Shape { what, detail }) => format!("{what}: {detail}"),
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    #[test]
    fn check_split_holds_the_fold() {
        let (counts, s) = call();
        counts.check_split(&s).expect("the fold's own sums");
    }

    #[test]
    fn check_split_names_each_time_the_list_drops() {
        for (name, drop) in [
            (
                "card_out_ms",
                (|c: &mut LbCount| c.card_out_ms = None) as fn(&mut LbCount),
            ),
            ("union_ms", |c| c.union_ns = 0),
            ("wait_ms", |c| c.wait_ns = 0),
            ("card_in_ms", |c| c.card_in_ms = Some(0.0)),
        ] {
            let (mut counts, s) = call();
            drop(&mut counts.lbs[5]);
            let e = refused(&counts, &s);
            assert!(e.contains(name), "{name}: {e}");
        }
    }

    #[test]
    fn check_split_refuses_another_window() {
        let (counts, mut s) = call();
        s.batches = 4;
        let e = refused(&counts, &s);
        assert!(e.contains("once per call"), "{e}");
    }

    #[test]
    fn check_split_refuses_a_shadow_time_without_a_block() {
        let (mut counts, s) = call();
        counts.lbs[0].card_in_ms = Some(0.0);
        let e = refused(&counts, &s);
        assert!(e.contains("was not timed as the split was"), "{e}");
    }

    #[test]
    fn check_split_leaves_an_untimed_split() {
        let (counts, mut s) = call();
        s.card_timed = false;
        s.batches = 9;
        counts
            .check_split(&s)
            .expect("an untimed split is not held");
    }

    #[test]
    fn untimed_keeps_the_entries() {
        let (counts, _) = call();
        let u = counts.untimed();
        assert_eq!(u.lbs.len(), counts.lbs.len());
        for (a, b) in u.lbs.iter().zip(&counts.lbs) {
            assert_eq!(
                (a.batch, a.layer, a.block, a.route, a.shadow),
                (b.batch, b.layer, b.block, b.route, b.shadow)
            );
            assert_eq!(
                (a.union_ns, a.wait_ns, a.card_out_ms, a.card_in_ms),
                (0, 0, None, None)
            );
        }
    }
}
