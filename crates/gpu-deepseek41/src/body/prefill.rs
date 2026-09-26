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
//! A batch runs its layers in order, and each layer over the batch's chunks —
//! runs of at most [`CHUNK`] positions, cut at multiples of [`CHUNK`] so a
//! chunk's latent rows fill one staging buffer from the row of its first
//! position on — from the first chunk the layer needs:
//!
//! 1. before the first layer, once per batch: each chunk planned in order
//!    (the host step plan, after the tokens before it), every token's rows
//!    read at once ([`PromptRows`]), each chunk's image built and every image
//!    copied to the card in one transfer with every token's rope tables
//!    ([`AttnBatch::stage_tables`]), the attention piece's gather of each
//!    chunk's words into its own row, the embedding broadcast;
//! 2. per layer: the engram step where the layer carries a site, chunk by
//!    chunk; the attention sub-layer ([`AttnChain::enqueue_batch_layer`]):
//!    its projections once over sub-blocks of chunks, and chunk by chunk in
//!    position order only what carries state from a position to the next — a
//!    chunk of the block's latent rows staged and committed to the ring
//!    before the next chunk attends, a chunk before the block's first its
//!    latent rows alone; the MoE sub-layer over the block's
//!    chunks in its batch phases ([`crate::chain::ffn::FfnBatch`]) — the
//!    route of every chunk, the handoffs to the host, the card's shadow work
//!    of every chunk while one union call serves the layer's host experts for
//!    every token of the block, the sums back and one join; where the next
//!    layer is tapped and the call hands features over, the tap of the
//!    tokens whose rows are kept, one launch;
//! 3. after the last layer, only in the batch that holds the prompt's last
//!    position: the head, for that position alone.
//!
//! Every launch writes, per token, what its one-token launch writes, and the
//! ops that carry state from a position to the next (the ring, the
//! compressor's pooling, each token's visible counts) run in position order,
//! so the batch is the steps' numbers, not an approximation of them. The
//! batch's buffers ([`Batch`]) are made by the first batch, or before it by
//! [`prepare_prefill`], never per batch; the decode step's buffers and
//! launches do not change.
//!
//! A call that fails is taken back ([`GpuModel::rollback`]) to where it
//! found the model, unless a fault poisoned it: the fault is the model's
//! until a reset.

use std::ops::Range;
use std::time::Instant;

use cuda_core::{CudaEvent, sys};
use model::moe::UNION_MAX_COLS;

use bloomery_gpu::COL_GROUP;

use super::ced::Mode;
use super::*;
use crate::chain::attn::{AttnBatch, BatchIo, ChunkCaches, ChunkSource};
use crate::chain::ffn::{BlockIo, CardExperts, FfnBatch, JoinIo};
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
    let runs = batches(first, ids.len());
    let starts: Vec<usize> = runs.iter().map(|r| r.start).collect();
    let (window, mut sink) = match features {
        Some(f) => (Some(f.window), Some(f.sink)),
        None => (None, None),
    };
    {
        let (_, _, body) = m.body_parts(WHAT)?;
        body.begin_call(first, first + ids.len(), &starts, window)?;
    }
    let mut token = None;
    for r in runs {
        let run = &ids[r.start - first..r.end - first];
        let last = r.end == first + ids.len();
        let ran = m
            .run_rows(run.len(), WHAT, |gpu, w, body, head, pos| {
                body.enqueue_batch(
                    gpu,
                    w,
                    head,
                    BatchRun {
                        ids: run,
                        pos,
                        last,
                    },
                    observe,
                )
            })
            .and_then(|t| {
                if let Some(f) = sink.as_mut() {
                    let (gpu, _, body) = m.body_parts(WHAT)?;
                    if let Some((at, rows)) = body.batch_features(gpu, r.clone())? {
                        f(at, rows)?;
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
        missing: "the head of the batch that holds the prompt's last position",
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

/// The batch's buffers: see the module comment. Made once.
pub(super) struct Batch {
    /// The chunk images' host builder, the chunks' plans, every chunk's image
    /// laid end to end on the host and its card copy.
    image: StepImage,
    plans: Vec<StepPlan>,
    images: Vec<u32>,
    params: DeviceBuffer<u32>,
    /// The attention piece over images of [`CHUNK`] tokens, a row of words
    /// per chunk, and its batch-wide buffers.
    attn: AttnChain,
    proj: AttnBatch,
    glue: GlueBatch,
    ffn: FfnBatch,
    rows: PromptRows,
    /// Per token of the batch: the streams and the folds, ping and pong.
    hc: [DeviceBuffer<f32>; 2],
    folds: [DeviceBuffer<f32>; 2],
    /// Per chunk, per indexer layer: its list, [`CHUNK`] tokens of it.
    lists: Vec<Vec<DeviceBuffer<u32>>>,
    /// A chunk's latent rows before its commit, row `p % CHUNK` for position
    /// `p`.
    staging: DeviceTensor<u16>,
    /// With a feature tap: per token, its row, on the card and the host.
    taps: Option<DeviceBuffer<f32>>,
    taps_host: Vec<f32>,
    /// The host and card time of the batches since the last
    /// [`Body::take_prefill_stats`].
    stats: PrefillStats,
    /// Whether each layer's card work is timed by events ([`CARD_MARKS`] a
    /// layer, recorded only while this is on).
    card_timing: bool,
    card_marks: Vec<CudaEvent>,
    /// Per layer: whether the last batch ran its block, so its shadow pair
    /// was recorded.
    card_served: Vec<bool>,
}

/// Events a layer records while [`Body::set_prefill_card_timing`] is on:
/// before its first launch, where its route's copies to the host complete
/// (after its attention where it has no block), and after its shadow.
const CARD_MARKS: usize = 3;

/// Where a batch's time went, summed over the batches since the last
/// [`Body::take_prefill_stats`]. Host times are wall clock on the calling
/// thread; the card times are event pairs on the engine stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrefillStats {
    /// Batches run, and layer-batches that served the host tier.
    pub batches: u64,
    pub layer_batches: u64,
    /// The batch's host half before the launches ([`Body::plan_batch`]):
    /// the plans, the rows, the images and their copy, which waits for the
    /// stream.
    pub prologue_ns: u64,
    /// The launches' enqueue with every serve in it.
    pub chain_ns: u64,
    /// Of `chain_ns`: the union calls, the waits on the route's copies and
    /// the activations' copy into the union's view.
    pub union_ns: u64,
    pub wait_ns: u64,
    pub copy_ns: u64,
    /// Card time, with [`Body::set_prefill_card_timing`] on: from each
    /// layer's first launch to its route's copies (its attention alone where
    /// it has no block), and its shadow, which runs under the union; and the
    /// attention's batch-wide projection phases, summed.
    pub card_timed: bool,
    pub card_out_ms: f64,
    pub card_in_ms: f64,
    pub card_proj_ms: f64,
    /// Entries the batches put in the launch queue — launches, copies, event
    /// records, stream waits: those before each layer-batch's wait on its
    /// route's copies (the batch's first steps, the engram step, the
    /// attention, the route and its copies, the previous layer's upload,
    /// join and tap) and those of each shadow ([`Body::enqueue_batch`]'s
    /// counts). The attention's are counted as enqueued; the rest by the
    /// launches each site's code makes.
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
}

impl PrefillStats {
    /// One line of `key=value`: every sum in ms, and the host and card
    /// terms per layer-batch (`_lb`).
    #[must_use]
    pub fn describe(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        let per = |v: f64| {
            if self.layer_batches == 0 {
                0.0
            } else {
                v / self.layer_batches as f64
            }
        };
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
            "batches={} layer_batches={} prologue_ms={:.1} chain_ms={:.1} union_ms={:.1} \
             wait_ms={:.1} enqueue_ms={:.1} copy_ms={:.1} union_lb={:.2} wait_lb={:.2} \
             enqueue_lb={:.2} copy_lb={:.2} entries_route={:.1} entries_shadow={:.1} \
             excluded_lb={:.1} {card}",
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
            + self.hc.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.folds.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self
                .lists
                .iter()
                .flatten()
                .map(|b| b.num_bytes())
                .sum::<usize>()
            + self.staging.buf().num_bytes()
            + self.taps.as_ref().map_or(0, |t| t.num_bytes())
    }
}

/// One batch of a prompt: its ids, its first position, and whether it holds
/// the prompt's last position (the head's).
struct BatchRun<'a> {
    ids: &'a [u32],
    pos: u32,
    last: bool,
}

/// A layer's caches with each chunk's list ([`Batch::lists`]), as
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

    use crate::chain::ffn::CardExperts;

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

    /// The shadow of a block of `chunks` (`FfnPiece::enqueue_batch_shadow`)
    /// on the `experts` arm, with card experts or not: per chunk HC_PRE and
    /// the norm; per expert, the norm's q8_1 codes and scales copied into the
    /// block's planes when there are card experts; per slot, the card's
    /// gate·up, its q8_1 form and the down when there are, and the card sum;
    /// the shared expert's gate·up, its down's q8_1 form, the down and past
    /// one token its copy token-major. Per expert, then over the block: the
    /// buckets, the grouped gate·up, its q8_1 form and the grouped down when
    /// there are card experts, and the card sum; on the tile arm also the
    /// tile table and the gather into run order.
    pub(super) fn shadow(
        chunks: &[Range<usize>],
        experts: CardExperts,
        card: bool,
        shared: Shared,
    ) -> u64 {
        let grouped = !matches!(experts, CardExperts::Slot);
        let tile = matches!(experts, CardExperts::Tile);
        let per_chunk: u64 = chunks
            .iter()
            .map(|r| {
                let m = r.len() as u64;
                let routed = match (grouped, card) {
                    (true, true) => 2,
                    (true, false) => 0,
                    (false, true) => 3 + 1,
                    (false, false) => 1,
                };
                let gate_up = if shared.gate_up_q3k { 1 } else { m };
                2 + routed + gate_up + u64::from(shared.down_q8) + 1 + u64::from(m > 1)
            })
            .sum();
        let block = match (grouped, card) {
            // The buckets, the gate·up, its q8_1 form and the down, and the
            // card sum; the tile arm adds the tile table and the gather.
            (true, true) => 4 + 2 * u64::from(tile) + 1,
            (true, false) => 1,
            (false, _) => 0,
        };
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
    /// the batch layout, the glue's and the MoE sub-layer's scratch, the
    /// batch's streams, folds, lists, staging and images, the host union's
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
            CHUNKS_MAX,
        )?;
        if attn.top_k() != self.attn.top_k() {
            attn.set_top_k(gpu, self.attn.top_k())?;
        }
        let writers = self.lists.first().map_or(0, Vec::len);
        let list_len = attn.list_len();
        let n = hp.n_embd;
        let tap_width = self.tap.as_ref().map_or(0, FeatureTap::width);
        let params = DeviceBuffer::zeroed(stream, CHUNKS_MAX * words)?;
        let glue = self.glue.batch(gpu, image.layout())?;
        let ffn = FfnBatch::new(gpu, n, hp.experts.ff, T_MAX, CardExperts::from_env()?)?;
        let hc = [
            DeviceBuffer::zeroed(stream, T_MAX * HC_STREAMS * n)?,
            DeviceBuffer::zeroed(stream, T_MAX * HC_STREAMS * n)?,
        ];
        let folds = [
            DeviceBuffer::zeroed(stream, T_MAX * n)?,
            DeviceBuffer::zeroed(stream, T_MAX * n)?,
        ];
        let lists = (0..CHUNKS_MAX)
            .map(|_| {
                (0..writers)
                    .map(|_| DeviceBuffer::zeroed(stream, CHUNK * list_len))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let staging = DeviceTensor::zeroed(stream, CHUNK, hp.head_dim)?;
        let taps = (tap_width > 0)
            .then(|| DeviceBuffer::zeroed(stream, T_MAX * tap_width))
            .transpose()?;
        let card_marks = (0..CARD_MARKS * self.layers.len())
            .map(|_| {
                gpu.context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Last: its refusal names the card's free bytes with every other
        // buffer of the batch already taken.
        let proj = attn.batch(gpu, T_MAX)?;
        Ok(Batch {
            plans: vec![StepPlan::default(); CHUNKS_MAX],
            images: vec![0; CHUNKS_MAX * words],
            params,
            glue,
            ffn,
            rows: self.rows.prompt_rows(T_MAX),
            hc,
            folds,
            lists,
            staging,
            taps,
            taps_host: vec![0.0; T_MAX * tap_width],
            stats: PrefillStats::default(),
            card_timing: false,
            card_marks,
            card_served: vec![false; self.layers.len()],
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
            .map(|b| std::mem::take(&mut b.stats))
            .unwrap_or_default()
    }

    /// Time each layer's card work with events from the next batch on
    /// ([`PrefillStats`]); off, no event is recorded. Makes the batch's
    /// buffers.
    pub fn set_prefill_card_timing(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let batch = self.batch_mut(gpu)?;
        batch.card_timing = on;
        batch.proj.set_card_timing(gpu, on, layers)
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

    /// A prompt call of positions `first .. end`, fed as batches from
    /// `starts`, whose reader keeps the features of its last `window`
    /// positions when `window` is given: its needs ([`super::ced`]), kept for
    /// the batches and [`Body::prefill_need`], and its hole. Refused with a
    /// window and no tap, and on a card that does not run every layer.
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
        let need = self.ced.need(first, end, starts, taps);
        let hole = need.hole();
        if !hole.is_empty() {
            self.holes.push(hole);
        }
        self.need = Some(need);
        Ok(())
    }

    /// The feature rows the batch of positions `run` kept — those of its
    /// positions from the call's first kept one on, `None` when it holds
    /// none — and the first position they hold, copied to the host in one
    /// transfer (a blocking read on the engine stream). Refused without a
    /// tap.
    fn batch_features(
        &mut self,
        gpu: &Gpu,
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
        let dev = batch.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "a feature tap (attach_features)",
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

    /// One batch: the prompt's ids `ids` at positions `pos ..`, the head only
    /// when `last`. Returns whether it enqueued the head. See the module
    /// comment. A batch that is not the last reads the fault word back at its
    /// end (a blocking read) and returns the fault as the named error.
    fn enqueue_batch(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        run: BatchRun<'_>,
        observe: &mut BatchObserver<'_>,
    ) -> Result<bool, GpuError> {
        let BatchRun { ids, pos, last } = run;
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
        let (b, u) = (pos as usize, ids.len());
        if self.history.len() != b || u == 0 || u > T_MAX || b + u > self.positions() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a batch of {u} positions (1..={T_MAX}) at {pos} after {} tokens, in caches \
                     of {} positions",
                    self.history.len(),
                    self.positions()
                ),
            });
        }
        if self
            .need
            .as_ref()
            .is_none_or(|n| b < n.first || b + u > n.end)
        {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a call's needs that cover the batch (begin_call)",
            });
        }
        let _ = self.batch_mut(gpu)?;
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
        let cuts = chunks(b, u);
        let t0 = Instant::now();
        self.plan_batch(stream, ids, &cuts)?;
        let prologue = nanos(t0.elapsed());
        for p in b..b + u {
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
        self.enqueue_batch_chain(gpu, w, head, &cuts, last, observe)?;
        let chain = nanos(t1.elapsed());
        let st = self.hybrid.stats();
        let union = st.batch_ns.saturating_sub(union0);
        let excluded = st.batch_excluded_slots.saturating_sub(excluded0);
        self.account_batch(prologue, chain, union, excluded)?;
        if last {
            return Ok(true);
        }
        match gpu.fault()? {
            Some(fault) => Err(GpuError::Fault { what: WHAT, fault }),
            None => Ok(false),
        }
    }

    /// Add one batch's times and the host slots its services' exclusion
    /// skipped to the stats; with card timing on, read its layers' event
    /// pairs, which waits for the last of them.
    fn account_batch(
        &mut self,
        prologue: u64,
        chain: u64,
        union: u64,
        excluded: u64,
    ) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let batch = self.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        let serve = batch.ffn.take_serve_times();
        let st = &mut batch.stats;
        st.batches += 1;
        st.prologue_ns += prologue;
        st.chain_ns += chain;
        st.union_ns += union;
        st.wait_ns += serve.wait_ns;
        st.copy_ns += serve.copy_ns;
        st.excluded_slots += excluded;
        if !batch.card_timing {
            return Ok(());
        }
        st.card_timed = true;
        st.card_proj_ms += batch.proj.take_card_ms()?;
        let (marks, _) = batch.card_marks.as_chunks::<CARD_MARKS>();
        for ([e0, e1, e2], &served) in marks.iter().zip(&batch.card_served).take(layers) {
            st.card_out_ms += f64::from(e0.elapsed_ms(e1)?);
            if served {
                st.card_in_ms += f64::from(e1.elapsed_ms(e2)?);
            }
        }
        Ok(())
    }

    /// The batch's host half: each chunk planned after the tokens before it
    /// (the ids joining the history), every token's rows read, each chunk's
    /// image built, and every image copied to the card in one transfer, which
    /// synchronizes the stream.
    fn plan_batch(
        &mut self,
        stream: &CudaStream,
        ids: &[u32],
        cuts: &[Range<usize>],
    ) -> Result<(), GpuError> {
        let Body {
            batch,
            planner,
            history,
            file,
            ..
        } = self;
        let batch = batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers",
        })?;
        let b = cuts.first().map_or(0, |r| r.start);
        let n = cuts.len();
        for (plan, r) in batch.plans.iter_mut().zip(cuts) {
            let toks = &ids[r.start - b..r.end - b];
            planner
                .plan_into(toks, r.start as u32, history, plan)
                .map_err(|e| GpuError::plan(WHAT, e))?;
            history.extend_from_slice(toks);
        }
        batch.rows.fill(file, &batch.plans[..n])?;
        let words = batch.image.layout().words();
        for (k, r) in cuts.iter().enumerate() {
            let at = r.start - b..r.end - b;
            batch.image.build(
                &batch.plans[k],
                batch.rows.embd(at.clone())?,
                batch.rows.engram(at.clone())?,
            )?;
            batch.images[k * words..(k + 1) * words].copy_from_slice(batch.image.words());
            batch.proj.stage_tables(
                batch.image.layout(),
                batch.image.words(),
                at.start,
                at.len(),
            )?;
        }
        let mut dst = span_mut(WHAT, &mut batch.params, 0, n * words)?;
        dst.copy_from_host(stream, &batch.images[..n * words])?;
        batch.proj.upload_tables(stream)
    }

    /// The batch's launches (the module comment's steps 1 to 3 on the card),
    /// each layer over the chunks the call's needs give it, the host tier
    /// served layer by layer while they are enqueued.
    fn enqueue_batch_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        cuts: &[Range<usize>],
        last: bool,
        observe: &mut BatchObserver<'_>,
    ) -> Result<(), GpuError> {
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
        let n = hp.n_embd;
        let words = batch.image.layout().words();
        let b = cuts.first().map_or(0, |r| r.start);
        let u: usize = cuts.iter().map(Range::len).sum();
        let s4 = HC_STREAMS * n;
        // Queue entries before each layer-batch's wait, and of its shadow.
        let (mut route, mut shadow) = (0u64, 0u64);
        for k in 0..cuts.len() {
            let p = span(WHAT, &batch.params, k * words, words)?;
            batch.attn.enqueue_step_of(gpu, &p, k)?;
        }
        route += queue::GATHER * cuts.len() as u64 + queue::embed(u);
        for (k, r) in cuts.iter().enumerate() {
            let (at, m) = (r.start - b, r.len());
            let p = span(WHAT, &batch.params, k * words, words)?;
            let [h0, _] = &mut batch.hc;
            let [f0, _] = &mut batch.folds;
            let mut s = span_mut(WHAT, h0, at * s4, m * s4)?;
            let mut f = span_mut(WHAT, f0, at * n, m * n)?;
            glue.enqueue_batch_embed(gpu, &batch.glue, &p, m, &mut s, &mut f)?;
        }
        // The batch's token where the chunk `k` on starts, `u` past the last.
        let token = |k: usize| cuts.get(k).map_or(u, |r| r.start - b);
        let mut cur = Cursor::default();
        let stream = gpu.stream();
        for (i, l) in layers.clone().enumerate() {
            let step = steps[i];
            let timed = batch.card_timing;
            if timed {
                batch.card_marks[CARD_MARKS * i].record(stream)?;
                route += 1;
            }
            batch.card_served[i] = false;
            // The layer runs a suffix of the chunks: its latent part from
            // `run`, its block from `full`.
            let run = cuts
                .iter()
                .position(|r| need.mode(i, r.start) != Mode::None)
                .unwrap_or(cuts.len());
            let full = cuts
                .iter()
                .position(|r| need.mode(i, r.start) == Mode::Full)
                .unwrap_or(cuts.len());
            if step.engram {
                let wkv_q8 = queue::q8(w, &names::engram_wkv(l));
                for (k, r) in cuts.iter().enumerate().skip(run) {
                    let (at, m) = (r.start - b, r.len());
                    route += queue::engram(m, wkv_q8);
                    let p = span(WHAT, &batch.params, k * words, words)?;
                    let (hin, hout) = ping(&mut batch.hc, cur.s);
                    let streams = span(WHAT, hin, at * s4, m * s4)?;
                    let mut out = span_mut(WHAT, hout, at * s4, m * s4)?;
                    let pre = span(WHAT, batch.ffn.hc(), at * HC_MIX, m * HC_MIX)?;
                    let mut input = span_mut(WHAT, &mut batch.folds[cur.f], at * n, m * n)?;
                    glue.enqueue_batch_engram(
                        gpu,
                        w,
                        &mut batch.glue,
                        l,
                        &p,
                        m,
                        EngramStep {
                            streams: &streams,
                            pre: &pre,
                            out: &mut out,
                            input: &mut input,
                        },
                    )?;
                }
                cur.s ^= 1;
                observe(
                    gpu,
                    BatchSeam {
                        kind: BatchSeamKind::Engram,
                        layer: l,
                        first: (b + token(run)) as u32,
                        at: token(run),
                        tokens: u - token(run),
                        streams: &batch.hc[cur.s],
                        fold: Some(&batch.folds[cur.f]),
                        attn: None,
                    },
                )?;
            }
            {
                let (sin, sout) = ping(&mut batch.hc, cur.s);
                let (fin, fout) = ping(&mut batch.folds, cur.f);
                let mut caches = LayerCaches {
                    kv: &mut kv[..],
                    shadows: &mut *shadows,
                    lists: &mut batch.lists[..],
                    i,
                    step,
                };
                batch.attn.enqueue_batch_layer(
                    gpu,
                    w,
                    l,
                    &mut batch.proj,
                    BatchIo {
                        cuts,
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
            route += batch.proj.take_entries();
            cur.s ^= 1;
            cur.f ^= 1;
            let at = token(full);
            observe(
                gpu,
                BatchSeam {
                    kind: BatchSeamKind::Attn,
                    layer: l,
                    first: (b + at) as u32,
                    at,
                    tokens: u - at,
                    streams: &batch.hc[cur.s],
                    fold: Some(&batch.folds[cur.f]),
                    attn: Some(batch.attn.batch_taps(&batch.proj)),
                },
            )?;
            if full == cuts.len() && timed {
                batch.card_marks[CARD_MARKS * i + 1].record(stream)?;
                route += 1;
            }
            if full < cuts.len() {
                batch.card_served[i] = true;
                batch.stats.layer_batches += 1;
                let block = BlockIo {
                    chunks: &cuts[full..],
                    base: b,
                    streams: &batch.hc[cur.s],
                    fold_in: &batch.folds[cur.f],
                };
                let bl = ffn.resolve_batch(w, l)?;
                ffn.enqueue_batch_route(gpu, &bl, &mut batch.ffn, &block, slots)?;
                batch.ffn.enqueue_download(gpu, at, u)?;
                route += queue::route(cuts.len() - full);
                if timed {
                    batch.card_marks[CARD_MARKS * i + 1].record(stream)?;
                    route += 1;
                }
                let card = CardStacks::of(w, l)?;
                let block = BlockIo {
                    chunks: &cuts[full..],
                    base: b,
                    streams: &batch.hc[cur.s],
                    fold_in: &batch.folds[cur.f],
                };
                shadow += queue::shadow(
                    &cuts[full..],
                    batch.ffn.experts(),
                    card.is_some(),
                    queue::Shared::of(w, l),
                );
                ffn.enqueue_batch_shadow(gpu, &bl, card, &mut batch.ffn, &block)?;
                if timed {
                    batch.card_marks[CARD_MARKS * i + 2].record(stream)?;
                    shadow += 1;
                }
                batch.ffn.serve(hybrid, l, at, u)?;
                batch.ffn.enqueue_upload(gpu, at, u)?;
                route += queue::UPLOAD_JOIN;
                let (sin, sout) = ping(&mut batch.hc, cur.s);
                let (_, fout) = ping(&mut batch.folds, cur.f);
                ffn.enqueue_batch_join(
                    gpu,
                    &batch.ffn,
                    l,
                    at,
                    u,
                    JoinIo {
                        streams: sin,
                        streams_out: sout,
                        fold_out: step.folds.then_some(fout),
                    },
                )?;
            }
            cur.s ^= 1;
            if step.folds {
                cur.f ^= 1;
            }
            observe(
                gpu,
                BatchSeam {
                    kind: BatchSeamKind::Ffn,
                    layer: l,
                    first: (b + at) as u32,
                    at,
                    tokens: u - at,
                    streams: &batch.hc[cur.s],
                    fold: step.folds.then_some(&batch.folds[cur.f]),
                    attn: None,
                },
            )?;
            // The tap of the kept rows: every one of them is in the block.
            let kept = need.features.max(b) - b;
            if kept < u
                && let (Some(tap), Some(dev)) = (tap.as_ref(), batch.taps.as_mut())
                && let Some(slot) = tap.after.get(i).copied().flatten()
            {
                let width = tap.width();
                let m = u - kept;
                let s = span(WHAT, &batch.hc[cur.s], kept * s4, m * s4)?;
                let mut rows = span_mut(WHAT, dev, kept * width, m * width)?;
                batch
                    .ffn
                    .enqueue_tap_means(gpu, &s, m, width, slot * n, &mut rows)?;
                route += 1;
            }
        }
        batch.stats.entries_route += route;
        batch.stats.entries_shadow += shadow;
        if last {
            let t = u - 1;
            let s = span(WHAT, &batch.hc[cur.s], t * s4, s4)?;
            let pre = span(WHAT, batch.ffn.hc(), t * HC_MIX, HC_MIX)?;
            glue.enqueue_head(gpu, w, &s, &pre, head)?;
        }
        Ok(())
    }
}
