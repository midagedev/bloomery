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
//! Every launch writes, per token, what its one-token launch writes, and the
//! ops that carry state from a position to the next (the ring, the
//! compressor's pooling, each token's visible counts) run in position order,
//! so the batch is the steps' numbers, not an approximation of them. What a
//! layer-batch leaves for a later layer of the same batch — the streams and
//! folds, the lists, the HC_PRE results, the images and tables — is kept per
//! batch of the group; what it consumes before the next layer-batch's
//! launches write it is one buffer, since the stream runs them in order; the
//! host copies come in two sets, since the host reads them outside that
//! order. The batch's buffers ([`Batch`]) are made by the first group, or
//! before it by [`prepare_prefill`], never per group; the decode step's
//! buffers and launches do not change.
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
use model::moe::UNION_MAX_COLS;

use bloomery_gpu::COL_GROUP;

use super::ced::Mode;
use super::*;
use crate::chain::attn::{AttnBatch, BatchIo, ChunkCaches, ChunkSource};
use crate::chain::ffn::{BatchLayer, BlockIo, ExchangeKey, FfnBatch, JoinIo};
use crate::chain::glue::{GlueBatch, PromptRows};
use crate::chain::nanos;
use crate::hc::{HC_MAX_TOKENS, HC_MIX};
use crate::span::{span, span_mut};

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = UNION_MAX_COLS;

/// Positions one chunk runs at most: the m-column kernels' and HC_PRE's.
pub const CHUNK: usize = HC_MAX_TOKENS;
// A chunk is a column group of the batch-wide projections.
const _: () = assert!(CHUNK == COL_GROUP);

/// Chunks a batch cuts into at most: an unaligned first position adds one.
const CHUNKS_MAX: usize = T_MAX / CHUNK + 1;

const WHAT: &str = "deepseek41 prefill";

/// How a prompt is fed ([`PrefillMode::from_env`]): in batches ([`prefill`])
/// or one decode step per id — the same-binary timing arm, which is the
/// decode step and not a second implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Batch,
    Steps,
}

impl PrefillMode {
    /// `BLOOMERY_PREFILL`: unset or `batch` batches, `steps` steps; any other
    /// value is refused by name.
    pub fn from_env() -> Result<PrefillMode, GpuError> {
        match std::env::var("BLOOMERY_PREFILL").as_deref() {
            Err(_) | Ok("batch") => Ok(PrefillMode::Batch),
            Ok("steps") => Ok(PrefillMode::Steps),
            Ok(_) => Err(GpuError::State {
                what: "BLOOMERY_PREFILL",
                missing: "batch or steps",
            }),
        }
    }

    /// The name a `load` line prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PrefillMode::Batch => "batch",
            PrefillMode::Steps => "steps",
        }
    }
}

/// Make the batch's buffers now, so the first [`prefill`] allocates nothing:
/// a timed prompt then times the batch alone. Idempotent.
pub fn prepare_prefill(m: &mut Deepseek41Model) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.batch_mut(gpu).map(|_| ())
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
        let group = body.batch_mut(gpu)?.group;
        body.begin_call(first, end, &starts, window)?;
        group
    };
    let mut token = None;
    for g in groups(runs.len(), group) {
        let rs = &runs[g];
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
            Err(e) => return Err(take_back(m, first, e)),
        }
    }
    token.ok_or(GpuError::State {
        what: WHAT,
        missing: "the head of the group that holds the prompt's last position",
    })
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

/// `BLOOMERY_PREFILL_GROUP`: batches a group holds, read when the batch's
/// buffers are made — unset 2, at most [`GROUP_MAX`]; any other value is
/// refused by name. 1 runs each batch alone, every layer of it before the
/// next batch's first.
pub(super) fn group_lever() -> Result<usize, GpuError> {
    let refused = GpuError::State {
        what: "BLOOMERY_PREFILL_GROUP",
        missing: "a whole number of batches from 1 to 8",
    };
    match std::env::var("BLOOMERY_PREFILL_GROUP") {
        Err(std::env::VarError::NotPresent) => Ok(2),
        Err(std::env::VarError::NotUnicode(_)) => Err(refused),
        Ok(v) => match v.parse::<usize>() {
            Ok(g) if (1..=GROUP_MAX).contains(&g) => Ok(g),
            _ => Err(refused),
        },
    }
}

/// The largest `BLOOMERY_PREFILL_GROUP`.
const GROUP_MAX: usize = 8;
// The refusal names the range.
const _: () = assert!(GROUP_MAX == 8);

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
    /// records, stream waits: those of each layer-batch's route (the batch's
    /// first steps, the engram step, the attention, the route and its
    /// copies) with those of its upload, join and tap, and those of each
    /// shadow ([`Body::enqueue_group`]'s counts). The attention's are counted
    /// as enqueued; the rest by the launches each site's code makes.
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

impl PrefillStats {
    /// One line of `key=value`: every sum in ms, and the host and card
    /// terms per layer-batch (`_lb`); `wait_first_lb` per layer-batch of a
    /// group's first batch.
    #[must_use]
    pub fn describe(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        let over = |v: f64, n: u64| if n == 0 { 0.0 } else { v / n as f64 };
        let per = |v: f64| over(v, self.layer_batches);
        let card = if self.card_timed {
            format!(
                "card_out_ms={:.1} card_in_ms={:.1} card_proj_ms={:.1} card_out_lb={:.2} \
                 card_in_lb={:.2} card_proj_lb={:.2}",
                self.card_out_ms,
                self.card_in_ms,
                self.card_proj_ms,
                per(self.card_out_ms),
                per(self.card_in_ms),
                per(self.card_proj_ms)
            )
        } else {
            "card=untimed".to_string()
        };
        format!(
            "group={} batches={} layer_batches={} prologue_ms={:.1} chain_ms={:.1} \
             union_ms={:.1} wait_ms={:.1} enqueue_ms={:.1} copy_ms={:.1} union_lb={:.2} \
             wait_lb={:.2} wait_first_lb={:.2} enqueue_lb={:.2} copy_lb={:.2} \
             entries_route={:.1} entries_shadow={:.1} excluded_lb={:.1} {card}",
            self.group,
            self.batches,
            self.layer_batches,
            ms(self.prologue_ns),
            ms(self.chain_ns),
            ms(self.union_ns),
            ms(self.wait_ns),
            ms(self.enqueue_ns()),
            ms(self.copy_ns),
            per(ms(self.union_ns)),
            per(ms(self.wait_ns)),
            over(ms(self.wait_first_ns), self.first_layer_batches),
            per(ms(self.enqueue_ns())),
            per(ms(self.copy_ns)),
            per(self.entries_route as f64),
            per(self.entries_shadow as f64),
            per(self.excluded_slots as f64),
        )
    }
}

impl Batch {
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

/// One group of a prompt: its ids, its batches' positions, and whether it
/// holds the prompt's last position (the head's).
struct GroupRun<'a> {
    ids: &'a [u32],
    runs: &'a [Range<usize>],
    last: bool,
}

/// A batch of a group as its layers run: its set of buffers, its chunks, its first
/// position and tokens, and which half of its ping-pong pairs its next
/// sub-layer reads.
struct Member {
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

    /// The host exchange's key of model layer `layer`'s block of the batch,
    /// from its token `at` on.
    fn key(&self, layer: usize, at: usize) -> ExchangeKey {
        ExchangeKey {
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

/// Queue entries — launches, copies, event records, stream waits — of the
/// batch's launch sites outside the attention piece, by the launches each
/// site's code makes; the piece counts its own
/// ([`AttnBatch::take_entries`]).
mod queue {
    use std::ops::Range;

    use bloomery_gpu::weights::{DevWeight, Weights};
    use gguf::quant::GgmlType;
    use model::arch::deepseek41::names;

    /// A chunk's words gather (`AttnChain::enqueue_step_of`).
    pub(super) const GATHER: u64 = 1;
    /// After a layer-batch's union: the host sums' copy
    /// (`FfnBatch::enqueue_upload`) and the join (`enqueue_batch_join`).
    pub(super) const UPLOAD_JOIN: u64 = 2;

    /// Whether the resident weight `name` is read in q8_1: a Q3_K or Q4_K
    /// row stream.
    pub(super) fn q8(w: &Weights, name: &str) -> bool {
        matches!(
            w.get(name),
            Some(DevWeight::KQuant {
                ty: GgmlType::Q3_K | GgmlType::Q4_K,
                ..
            })
        )
    }

    /// The embedding broadcast of `tokens` tokens
    /// (`GlueBatch::enqueue_batch_embed`): a launch a token.
    pub(super) fn embed(tokens: usize) -> u64 {
        tokens as u64
    }

    /// A chunk of `m` tokens' engram step (`Glue::enqueue_batch_engram`): a
    /// row launch a token, the q8_1 form when `wkv` reads one, the wkv
    /// projection and past one token its copy token-major, the key norm, the
    /// gate and the fold.
    pub(super) fn engram(m: usize, wkv_q8: bool) -> u64 {
        m as u64 + u64::from(wkv_q8) + 1 + u64::from(m > 1) + 3
    }

    /// The route of a block of `chunks` chunks (`FfnPiece::enqueue_batch_route`
    /// and `FfnBatch::enqueue_download`): each chunk's norm, the router's two
    /// launches, the places, the three copies to the host and their event.
    pub(super) fn route(chunks: usize) -> u64 {
        chunks as u64 + 2 + 1 + 3 + 1
    }

    /// A layer's shared expert as resident: gate·up both Q3_K (one launch
    /// for a chunk, else one a token), and a down read in q8_1.
    #[derive(Clone, Copy)]
    pub(super) struct Shared {
        gate_up_q3k: bool,
        down_q8: bool,
    }

    impl Shared {
        pub(super) fn of(w: &Weights, l: usize) -> Shared {
            let q3k = |name: &str| {
                matches!(
                    w.get(name),
                    Some(DevWeight::KQuant {
                        ty: GgmlType::Q3_K,
                        ..
                    })
                )
            };
            Shared {
                gate_up_q3k: q3k(&names::ffn_gate_shexp(l)) && q3k(&names::ffn_up_shexp(l)),
                down_q8: q8(w, &names::ffn_down_shexp(l)),
            }
        }
    }

    /// The shadow of a block of `chunks` (`FfnPiece::enqueue_batch_shadow`),
    /// with card experts or not: per chunk HC_PRE and the norm, the norm's
    /// q8_1 codes and scales copied into the block's planes when there are
    /// card experts, the shared expert's gate·up, its down's q8_1 form, the
    /// down and past one token its copy token-major; then over the block the
    /// buckets, the tile table, the gather into run order, the gate·up, its
    /// q8_1 form and the down when there are card experts, and the card sum.
    pub(super) fn shadow(chunks: &[Range<usize>], card: bool, shared: Shared) -> u64 {
        let per_chunk: u64 = chunks
            .iter()
            .map(|r| {
                let m = r.len() as u64;
                let routed = 2 * u64::from(card);
                let gate_up = if shared.gate_up_q3k { 1 } else { m };
                2 + routed + gate_up + u64::from(shared.down_q8) + 1 + u64::from(m > 1)
            })
            .sum();
        // The buckets, the tile table, the gather, the gate·up, its q8_1 form
        // and the down, and the card sum.
        let block = if card { 7 } else { 1 };
        per_chunk + block
    }
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
    /// The batch's buffers, made by the first call: the attention piece over
    /// the batch layout, the glue's and the MoE sub-layer's scratch, each
    /// batch's streams, folds, lists and images for a group of the most
    /// batches `BLOOMERY_PREFILL_GROUP` gives, the staging, the host union's
    /// scratch.
    fn batch_mut(&mut self, gpu: &Gpu) -> Result<&mut Batch, GpuError> {
        if self.batch.is_none() {
            let b = self.make_batch(gpu)?;
            self.hybrid.host_mut().prepare_union()?;
            self.batch = Some(Box::new(b));
        }
        self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })
    }

    fn make_batch(&self, gpu: &Gpu) -> Result<Batch, GpuError> {
        let hp = &self.hp;
        let stream = gpu.stream();
        let group = group_lever()?;
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
        let ffn = FfnBatch::new(gpu, n, hp.experts.ff, [T_MAX, sets])?;
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
                    taps: (tap_width > 0).then(|| z(T_MAX * tap_width)).transpose()?,
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
            image,
            attn,
            proj,
        })
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
    /// ([`PrefillStats`]); off, no event is recorded. Makes the batch's
    /// buffers.
    pub fn set_prefill_card_timing(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let batch = self.batch_mut(gpu)?;
        batch.card_timing = on;
        let layer_batches = layers * batch.sets.len();
        batch.proj.set_card_timing(gpu, on, layer_batches)
    }

    /// Device bytes of the batch's buffers; 0 before the first batch.
    #[must_use]
    pub fn batch_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.device_bytes())
    }

    /// The part of [`Body::batch_bytes`] the batch-wide attention
    /// projections hold ([`crate::chain::attn::AttnBatch`]); 0 before the
    /// first batch.
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

    /// `BLOOMERY_PREFILL_GROUP` as the next batch's buffers would read it:
    /// batches a group holds (unset 2, from 1 to 8); any other value is
    /// refused by name.
    pub fn prefill_group_lever() -> Result<usize, GpuError> {
        group_lever()
    }

    /// A prompt call of positions `first .. end`, fed as batches from
    /// `starts`, whose reader keeps the features of its last `window`
    /// positions when `window` is given: its needs ([`super::ced`]), kept for
    /// the batches and [`Body::prefill_need`]. Its hole is recorded when its
    /// first group has planned ([`Body::plan_group`]). Refused with a window
    /// and no tap, and on a card that does not run every layer.
    fn begin_call(
        &mut self,
        first: usize,
        end: usize,
        starts: &[usize],
        window: Option<usize>,
    ) -> Result<(), GpuError> {
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
            (Some(_), Some(tap)) => Some(tap.after.iter().map(Option::is_some).collect()),
        };
        let taps = after.as_deref().zip(window);
        self.need = Some(self.ced.need(first, end, starts, taps));
        Ok(())
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
        let batch = self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "a batch that ran",
        })?;
        let dev = batch
            .sets
            .get(set)
            .and_then(|s| s.taps.as_ref())
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "a feature tap (attach_features) on the group's batch",
            })?;
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
        let GroupRun { ids, runs, last } = run;
        let stream = gpu.stream();
        if capturing(stream)? {
            return Err(GpuError::State {
                what: WHAT,
                missing: "an eager stream: a batch is served while it is enqueued",
            });
        }
        if self.rows_failed {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a reset: an earlier step's engram rows failed after its launch, and \
                          the card ran that step on stale rows",
            });
        }
        self.arrive()?;
        self.rows.finish()?;
        let sets = self.batch_mut(gpu)?.sets.len();
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
        let (union0, excluded0) = {
            let st = self.hybrid.stats();
            (st.batch_ns, st.batch_excluded_slots)
        };
        let t1 = Instant::now();
        let chained = self.enqueue_group_chain(gpu, w, head, &mut members, last, observe);
        let chain = nanos(t1.elapsed());
        let mut sums = match chained {
            Ok(sums) => sums,
            Err(e) => return Err(fault_or(gpu, b, e)),
        };
        // Before anything else can fail: a fault the group raised is the
        // call's error, never the next call's.
        if let Some(fault) = gpu.fault()? {
            return Err(GpuError::fault(WHAT, fault));
        }
        sums.chain_ns = chain;
        let st = self.hybrid.stats();
        sums.union_ns = st.batch_ns.saturating_sub(union0);
        sums.excluded_slots = st.batch_excluded_slots.saturating_sub(excluded0);
        sums.prologue_ns = prologue;
        self.account_group(members.len(), sums)?;
        Ok(last)
    }

    /// Add one group's sums, its `g` batches and, with card timing on, its
    /// layer-batches' event pairs — read first, which waits for the last of
    /// them — to the stats all at once: a group that fails adds nothing.
    fn account_group(&mut self, g: usize, mut sums: PrefillStats) -> Result<(), GpuError> {
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
            for ([e0, e1, e2, e3], &served) in marks.iter().zip(&batch.card_served).take(layers * g)
            {
                sums.card_out_ms += f64::from(e0.elapsed_ms(e1)?);
                if served {
                    sums.card_in_ms += f64::from(e2.elapsed_ms(e3)?);
                }
            }
        }
        batch.stats.add(&sums);
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
    ) -> Result<PrefillStats, GpuError> {
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
        batch.ffn.begin_group();
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
            n: hp.n_embd,
            sums: PrefillStats::default(),
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
        let mut next = match items.first() {
            Some(&(i, j)) => Some(cx.route(&mut members[j], i, observe)?),
            None => None,
        };
        for (x, &(i, j)) in items.iter().enumerate() {
            let now = next.take().ok_or(GpuError::State {
                what: WHAT,
                missing: "the route of the layer-batch the group serves",
            })?;
            cx.shadow(&members[j], i, &now)?;
            if ahead && let Some(&(i2, j2)) = items.get(x + 1) {
                next = Some(cx.route(&mut members[j2], i2, observe)?);
            }
            cx.serve(&members[j], i, &now)?;
            cx.post(&mut members[j], i, &now, observe)?;
            if !ahead && let Some(&(i2, j2)) = items.get(x + 1) {
                next = Some(cx.route(&mut members[j2], i2, observe)?);
            }
        }
        let sums = cx.sums;
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
        Ok(sums)
    }
}

/// What a group's layer-batches enqueue through: the body's parts they read
/// and write, the batch's buffers, the call's needs, and the group's sums so
/// far — the queue entries, the layer-batches, the serves' waits — which
/// reach the stats only once the group has run ([`Body::account_group`]).
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
    n: usize,
    sums: PrefillStats,
}

impl<'a> GroupCx<'a> {
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
        }
        self.sums.entries_route += queue::GATHER * m.cuts.len() as u64 + queue::embed(m.u);
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
        let served = m.set * self.layers.len() + i;
        if timed {
            self.mark(m.set, i, 0)?;
            self.sums.entries_route += 1;
        }
        self.batch.card_served[served] = false;
        // The layer runs a suffix of the chunks: its latent part from
        // `run`, its block from `full`.
        let run = m
            .cuts
            .iter()
            .position(|r| self.need.mode(i, r.start) != Mode::None)
            .unwrap_or(m.cuts.len());
        let full = m
            .cuts
            .iter()
            .position(|r| self.need.mode(i, r.start) == Mode::Full)
            .unwrap_or(m.cuts.len());
        if step.engram {
            self.engram(m, l, run, observe)?;
        }
        let batch = &mut *self.batch;
        let row0 = m.set * CHUNKS_MAX;
        let (u, b) = (m.u, m.b);
        let own = set_of(&mut batch.sets, m.set)?;
        {
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
                },
                &mut caches,
            )?;
        }
        self.sums.entries_route += batch.proj.take_entries();
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
                self.sums.entries_route += 1;
            }
            return Ok(Routed {
                full,
                at,
                block: None,
            });
        }
        batch.card_served[served] = true;
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
        batch.ffn.enqueue_download(gpu, m.key(l, at))?;
        self.sums.entries_route += queue::route(m.cuts.len() - full);
        if timed {
            self.mark(m.set, i, 1)?;
            self.sums.entries_route += 1;
        }
        Ok(Routed {
            full,
            at,
            block: Some(bl),
        })
    }

    /// Model layer `l`'s engram step over the batch `m`'s chunks from chunk
    /// `run` on, chunk by chunk. Shows `observe` its seam.
    fn engram(
        &mut self,
        m: &mut Member,
        l: usize,
        run: usize,
        observe: &mut BatchObserver<'_>,
    ) -> Result<(), GpuError> {
        let (gpu, w, n) = (self.gpu, self.w, self.n);
        let batch = &mut *self.batch;
        let words = batch.image.layout().words();
        let s4 = HC_STREAMS * n;
        let row0 = m.set * CHUNKS_MAX;
        let (u, b) = (m.u, m.b);
        let own = set_of(&mut batch.sets, m.set)?;
        let wkv_q8 = queue::q8(w, &names::engram_wkv(l));
        for (k, r) in m.cuts.iter().enumerate().skip(run) {
            let (at, len) = (r.start - b, r.len());
            self.sums.entries_route += queue::engram(len, wkv_q8);
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
        if timed {
            self.mark(m.set, i, 2)?;
            self.sums.entries_shadow += 1;
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
        self.sums.entries_shadow +=
            queue::shadow(&m.cuts[r.full..], card.is_some(), queue::Shared::of(w, l));
        self.ffn
            .enqueue_batch_shadow(gpu, bl, card, &mut batch.ffn, &block)?;
        if timed {
            self.mark(m.set, i, 3)?;
            self.sums.entries_shadow += 1;
        }
        Ok(())
    }

    /// Layer index `i`'s host experts for the batch `m`'s block, where it
    /// has one: the wait on its route's copies and one union call.
    fn serve(&mut self, m: &Member, i: usize, r: &Routed<'_>) -> Result<(), GpuError> {
        if r.block.is_none() {
            return Ok(());
        }
        let l = self.layers.start + i;
        let times = self.batch.ffn.serve(&mut *self.hybrid, m.key(l, r.at))?;
        let st = &mut self.sums;
        st.wait_ns += times.wait_ns;
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
        let batch = &mut *self.batch;
        let own = set_of(&mut batch.sets, m.set)?;
        if r.block.is_some() {
            batch.ffn.enqueue_upload(gpu, m.key(l, r.at))?;
            self.sums.entries_route += queue::UPLOAD_JOIN;
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
            && let (Some(tap), Some(dev)) = (self.tap, own.taps.as_mut())
            && let Some(slot) = tap.after.get(i).copied().flatten()
        {
            let width = tap.width();
            let k = m.u - kept;
            let s = span(WHAT, &own.hc[m.cur.s], kept * s4, k * s4)?;
            let mut rows = span_mut(WHAT, dev, kept * width, k * width)?;
            batch
                .ffn
                .enqueue_tap_means(gpu, &s, k, width, slot * n, &mut rows)?;
            self.sums.entries_route += 1;
        }
        Ok(())
    }
}

/// The error of a group from position `b` that failed with `e` after its
/// first launch: the stream waited for, then the fault word read — a fault
/// any of the group's launches raised is the error whatever `e` was, with
/// `e` behind it, so it poisons the model, never reaches a later call, and
/// keeps `e`'s text (a host refusal's layer, say). A fault `e` that names
/// the same fault stays as it is: it names the reader that met it first. A
/// failed wait or read names `e` beside its own error, and a fault `e`
/// stays.
fn fault_or(gpu: &Gpu, b: usize, e: GpuError) -> GpuError {
    let read = gpu
        .stream()
        .synchronize()
        .map_err(GpuError::from)
        .and_then(|()| gpu.fault());
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

/// The group's batch `set`'s own buffers, refused past the group.
fn set_of(sets: &mut [BatchSet], set: usize) -> Result<&mut BatchSet, GpuError> {
    let n = sets.len();
    sets.get_mut(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("batch {set} of a group of at most {n}"),
    })
}
