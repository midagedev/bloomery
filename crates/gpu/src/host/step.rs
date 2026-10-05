//! The step port: the handoff inside the captured chain. The card side is
//! the [`Boundary`] — the handoff region the router's launches write, the
//! host-mapped page ([`PageLayout`]) and the sum the combine reads — and the
//! host side is [`StepPort`], which serves each go in go order: waits for it
//! with the whole pool spinning on the generation word, checks the handoff's
//! sequence number and layer, has the model's host computation write the sum
//! into the page and adds one to the row's counter.

use super::page::{HandoffLayout, PageLayout, Word};
use super::route_trace::RouteTrace;
use super::slots::{MAX_TIERS, Slot, SlotMap};
use super::{Health, HostExperts, Refusal, nanos, non_finite, unknown_id};
use crate::GpuError;
use crate::graph::{
    MappedHost, capturing, mem_batch, op_add, op_barrier_sys, op_wait_geq, op_write,
};
use crate::tensor::window;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use model::ops::DEFER_MAX_COLS;
use model::{Tensor2, Tensor2View};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// What the counter is set to when a service fails: every wait still pending
/// in the stream passes, so a failed step drains instead of hanging.
pub const RELEASE: u32 = 1 << 30;

/// How long a service waits for its go before it gives up and releases the
/// stream.
pub(super) const GO_DEADLINE: Duration = Duration::from_secs(10);

/// Spins between two clock reads while waiting for a go.
const DEADLINE_POLL: u32 = 1 << 12;

/// The name every step service's errors and refusals carry.
pub(super) const SERVE: &str = "Hybrid::serve";

/// Flag word `w` of the page, as an atomic.
pub(super) fn word(page: &MappedHost, w: Word) -> &AtomicU32 {
    page.atomic_u32(w.offset())
        .expect("every flag word lies in the page's first PAYLOAD_OFF bytes")
}

/// Word `i` of the handoff image of `words` words at byte `at` of the page;
/// `None` past it.
fn payload_word(page: &MappedHost, at: usize, i: usize, words: usize) -> Option<u32> {
    if i >= words {
        return None;
    }
    // SAFETY: i < words, and an image's `words` words start at a word-aligned
    // `at` inside the page (`PageLayout` lays the images out, `Boundary::new`
    // sizes the page by it). The write that filled them finished before the
    // go the caller acquired, and none writes them again before the caller's
    // signal.
    Some(unsafe { page.host_at(at + 4 * i).cast::<u32>().read_volatile() })
}

/// Copy `dst.len()` f32 of the image of `words` words at byte `at` of the
/// page from word `from` into `dst`; `false` when the span passes the image.
fn payload_f32_into(
    page: &MappedHost,
    at: usize,
    from: usize,
    words: usize,
    dst: &mut [f32],
) -> bool {
    if from.checked_add(dst.len()).is_none_or(|end| end > words) {
        return false;
    }
    // SAFETY: the span is inside the image (checked above), which is
    // f32-aligned at `at` + 4·from; `dst` is a distinct host slice. Ordered
    // after the go as in `payload_word`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            page.host_at(at + 4 * from).cast::<f32>(),
            dst.as_mut_ptr(),
            dst.len(),
        );
    }
    true
}

/// Whether every operation enqueued on `stream` has finished, without
/// waiting for any.
pub(super) fn stream_idle(stream: &CudaStream) -> Result<bool, GpuError> {
    // SAFETY: the stream is live; the call only queries it.
    let rc = unsafe { sys::cuStreamQuery(stream.cu_stream()) };
    if rc == sys::cudaError_enum_CUDA_ERROR_NOT_READY {
        return Ok(false);
    }
    crate::graph::cu(rc, "cuStreamQuery")?;
    Ok(true)
}

/// Serial-number order on a wrapping u32: `a` comes strictly before `b`.
fn before(a: u32, b: u32) -> bool {
    a != b && b.wrapping_sub(a) < 1 << 31
}

// --------------------------------------------------------------- boundary

/// The shapes a boundary is cut for.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryShape {
    /// The model width: the normed activation, the host sum.
    pub hidden: usize,
    /// Routed slots per token.
    pub n_used: usize,
}

/// One row's part of the host-mapped page: its handoff image and the host
/// sum its combine reads, with their windows.
pub(crate) struct RowPage {
    /// Byte offsets of the image and of the sum in the page.
    image_off: usize,
    hsum_off: usize,
    image: ManuallyDrop<DeviceBuffer<u32>>,
    /// The row's host experts' weighted sum, read in place through the
    /// mapping.
    pub(crate) hsum: ManuallyDrop<DeviceBuffer<f32>>,
}

/// The card side of the boundary, allocated once at load (decision 4 of
/// docs/gpu-design.md: a captured graph bakes every address in): the handoff
/// region the router's launches write, the host-mapped page, and the sum the
/// combine reads.
///
/// A boundary of several rows lets that many tokens' handoffs be in flight
/// at once, served in go order: each row has its own counter, layer word,
/// image and sum in the page; the region, the generation and the sequence
/// are shared — the region's readers of one row finish on the stream before
/// the next row's norm writes it, and the other two count every row's goes
/// in one order, the order the host serves them in.
pub struct Boundary {
    /// Windows into `region` — what the norm and the router write in place
    /// of the MoE arena's own buffers on a hybrid layer.
    pub(crate) normed: ManuallyDrop<DeviceBuffer<f32>>,
    pub(crate) ids: ManuallyDrop<DeviceBuffer<u32>>,
    pub(crate) weights: ManuallyDrop<DeviceBuffer<f32>>,
    /// The region's sequence word: what a chain's own launch copies into a
    /// row's image in place of the copy node ([`Boundary::handoff_target`]).
    seq: ManuallyDrop<DeviceBuffer<u32>>,
    /// Per row, its image and sum in the page; never empty. A one-row
    /// chain reads row 0's sum as `pages[0].hsum`.
    pub(crate) pages: Vec<RowPage>,
    /// `shared expert + hsum`: the combine's shared-expert input.
    pub(crate) sum: DeviceBuffer<f32>,
    pub(super) region: DeviceBuffer<u32>,
    pub(super) page: MappedHost,
    pub(super) layout: PageLayout,
}

/// What a chain's own handoff launch reads and writes
/// ([`Boundary::handoff_target`]).
pub struct HandoffTarget<'a> {
    /// The activation the norm wrote into the region, `hidden` f32.
    pub x: &'a DeviceBuffer<f32>,
    /// The region's sequence word: the image carries it as it stands before
    /// the go adds one.
    pub seq: &'a DeviceBuffer<u32>,
    /// The handoff's image in the host-mapped page, `layout.x + hidden`
    /// words; the launch writes every field of it.
    pub image: &'a mut DeviceBuffer<u32>,
    pub layout: HandoffLayout,
}

impl Drop for RowPage {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — the boundary's page frees its allocation after the rows.
        unsafe {
            drop(ManuallyDrop::take(&mut self.image).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.hsum).into_raw_parts());
        }
    }
}

impl Drop for Boundary {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — `region` frees the allocation after this.
        unsafe {
            drop(ManuallyDrop::take(&mut self.normed).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.ids).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.weights).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.seq).into_raw_parts());
        }
    }
}

impl Boundary {
    /// Allocate the region, the page and the sum for `shape`. One row.
    /// Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        shape: BoundaryShape,
    ) -> Result<Boundary, GpuError> {
        Boundary::with_rows(ctx, stream, shape, 1)
    }

    /// [`Boundary::new`] with `rows` rows (1..=[`super::page::MAX_ROWS`]) of
    /// one column, laid out by [`PageLayout`]: row 0's image and sum sit where
    /// a one-row boundary has them, the other rows' images after it and every
    /// sum after the images, each at a 256-byte boundary. A shape the page
    /// cannot carry is refused by the bound it breaks. Load-time only.
    pub fn with_rows(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        shape: BoundaryShape,
        rows: usize,
    ) -> Result<Boundary, GpuError> {
        Boundary::with_cols(ctx, stream, shape, rows, 1)
    }

    /// [`Boundary::with_rows`] with `cols` columns a row
    /// (1..=`DEFER_MAX_COLS`): a row's image carries up to `cols` columns'
    /// routing and activations, its sum `cols` columns, and the activation
    /// window the norm writes is `cols · hidden` wide. A one-column chain
    /// uses column 0 of it. Load-time only.
    pub fn with_cols(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        shape: BoundaryShape,
        rows: usize,
        cols: usize,
    ) -> Result<Boundary, GpuError> {
        let what = "Boundary::new";
        let layout = PageLayout::new(rows, cols, shape.hidden, shape.n_used)
            .map_err(|e| GpuError::shape(what, e.to_string()))?;
        let bytes = layout
            .bytes()
            .ok_or_else(|| GpuError::shape(what, "the page's size passes usize"))?;
        let h = layout.handoff();
        let words = layout.image_words();
        let region = DeviceBuffer::<u32>::zeroed(stream, words)?;
        let page = MappedHost::new(ctx, bytes, "cuMemHostAlloc (hybrid page)")?;
        let base = region.cu_deviceptr();
        // SAFETY: the ids span is n_used words at `h.ids`, inside the region's
        // `words` words (`PageLayout` keeps the fields apart and before the
        // activation); `region` moves into the boundary beside the window (a
        // move of the handle, not of the allocation), where it outlives it
        // and is never reallocated.
        let ids = unsafe { window::<u32>(base + 4 * h.ids as u64, h.n_used, ctx) };
        // SAFETY: as above, for the weights' n_used words at `h.weights`.
        let weights = unsafe { window::<f32>(base + 4 * h.weights as u64, h.n_used, ctx) };
        // SAFETY: as above, for the activation's `cols · hidden` words at
        // `h.x`, the region's tail.
        let normed = unsafe { window::<f32>(base + 4 * h.x as u64, layout.hsum_len(), ctx) };
        // SAFETY: as for the ids, for the sequence's one word at `h.seq`.
        let seq = unsafe { window::<u32>(base + 4 * h.seq as u64, 1, ctx) };
        let pages = (0..rows)
            .map(|r| {
                let (image_off, hsum_off) = layout
                    .image_off(r)
                    .zip(layout.hsum_off(r))
                    .ok_or_else(|| GpuError::shape(what, format!("row {r} of the page")))?;
                // SAFETY: row r's image — `words` words from a 256-aligned
                // offset — and its sum — `cols · hidden` f32 from a 256-aligned
                // offset — lie inside the page (sized by the layout past the
                // last sum), apart from each other and from every other
                // row's; the page moves into the boundary beside the windows
                // and is freed only when the boundary drops.
                let (image, hsum) = unsafe {
                    (
                        window::<u32>(page.dev_at(image_off), words, ctx),
                        window::<f32>(page.dev_at(hsum_off), layout.hsum_len(), ctx),
                    )
                };
                Ok(RowPage {
                    image_off,
                    hsum_off,
                    image,
                    hsum,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Boundary {
            normed,
            ids,
            weights,
            seq,
            pages,
            sum: DeviceBuffer::<f32>::zeroed(stream, h.hidden)?,
            region,
            page,
            layout,
        })
    }

    /// Rows the boundary carries.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.pages.len()
    }

    /// The page's layout.
    #[must_use]
    pub fn layout(&self) -> PageLayout {
        self.layout
    }

    /// Row `row`'s part of the page; a row the boundary does not carry is
    /// refused.
    fn page_of(&self, row: usize, what: &'static str) -> Result<&RowPage, GpuError> {
        self.pages.get(row).ok_or_else(|| {
            GpuError::shape(
                what,
                format!("row {row} of a boundary of {} rows", self.pages.len()),
            )
        })
    }

    /// The handoff's activations, `cols · hidden` f32 (a one-column chain
    /// uses the first `hidden`): what the norm writes on a hybrid layer and
    /// the host reads.
    #[must_use]
    pub fn normed(&self) -> &DeviceBuffer<f32> {
        &self.normed
    }

    /// The handoff's activation, for the norm to write.
    pub fn normed_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.normed
    }

    /// The host experts' weighted sums, `cols · hidden` f32 in the
    /// host-mapped page: what a combine reads in place after the layer's
    /// wait. Row 0's.
    #[must_use]
    pub fn hsum(&self) -> &DeviceBuffer<f32> {
        &self.pages[0].hsum
    }

    /// Row `row`'s host sum ([`Boundary::hsum`]).
    pub fn hsum_of(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        Ok(&self.page_of(row, "Boundary::hsum_of")?.hsum)
    }

    /// What a chain's own launch writes the handoff through, in place of the
    /// copy node: the activation the norm wrote into the region, the region's
    /// sequence word, and the handoff's image in the host-mapped page with its
    /// layout. The launch copies the sequence word, the routing and the
    /// activation into the image; [`Boundary::enqueue_go`] follows it. Row
    /// 0's image.
    pub fn handoff_target(&mut self) -> HandoffTarget<'_> {
        self.handoff_target_of(0)
            .expect("every boundary carries row 0")
    }

    /// [`Boundary::handoff_target`] into row `row`'s image.
    pub fn handoff_target_of(&mut self, row: usize) -> Result<HandoffTarget<'_>, GpuError> {
        let rows = self.pages.len();
        let page = self.pages.get_mut(row).ok_or_else(|| {
            GpuError::shape(
                "Boundary::handoff_target_of",
                format!("row {row} of a boundary of {rows} rows"),
            )
        })?;
        Ok(HandoffTarget {
            x: &self.normed,
            seq: &self.seq,
            image: &mut page.image,
            layout: self.layout.handoff(),
        })
    }

    /// Bytes the handoff's copy node moves.
    pub(crate) fn handoff_bytes(&self) -> usize {
        4 * self.layout.image_words()
    }

    /// Device bytes the boundary holds: the region and the sum. The page is
    /// host memory.
    pub fn device_bytes(&self) -> usize {
        self.region.num_bytes() + self.sum.num_bytes()
    }

    /// Enqueue layer `layer`'s handoff: the region's image into row 0's
    /// image (one copy node), then the go batch ([`Boundary::enqueue_go`]).
    pub(crate) fn enqueue_out(&self, stream: &CudaStream, layer: usize) -> Result<(), GpuError> {
        let what = "Boundary::enqueue_out";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        let bytes = self.handoff_bytes();
        let image_off = self.page_of(0, what)?.image_off;
        // SAFETY: the page holds the region's image of `bytes` bytes at row
        // 0's image offset (the layout sizes it so); both allocations live as
        // long as the boundary, which outlives every graph that captured this
        // copy.
        let rc = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                self.page.host_at(image_off).cast(),
                self.region.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )
        };
        crate::graph::cu(rc, "cuMemcpyDtoHAsync_v2 (hybrid handoff)")?;
        self.go_batch(stream, layer, 0)
    }

    /// Enqueue layer `layer`'s go alone, for a chain whose own launch wrote the
    /// handoff's image ([`Boundary::handoff_target`]): a system barrier (the
    /// image lands first), the layer into the page, a second barrier, and one
    /// added to the generation word and to the sequence word the next handoff
    /// carries. The host serves it once the chain says the layer is enqueued
    /// ([`super::HostTier::layer_enqueued`]). Row 0's go.
    pub fn enqueue_go(&self, stream: &CudaStream, layer: usize) -> Result<(), GpuError> {
        self.enqueue_go_of(stream, layer, 0)
    }

    /// [`Boundary::enqueue_go`] of row `row`: its layer word takes the layer.
    pub fn enqueue_go_of(
        &self,
        stream: &CudaStream,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let what = "Boundary::enqueue_go";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        self.page_of(row, what)?;
        self.go_batch(stream, layer, row)
    }

    fn go_batch(&self, stream: &CudaStream, layer: u32, row: usize) -> Result<(), GpuError> {
        let seq = self.region.cu_deviceptr() + 4 * self.layout.handoff().seq as u64;
        mem_batch(
            stream,
            &mut [
                op_barrier_sys(),
                op_write(self.page.dev_at(Word::Lyr(row).offset()), layer),
                op_barrier_sys(),
                op_add(self.page.dev_at(Word::Gen.offset()), 1),
                op_add(seq, 1),
            ],
            "cuStreamBatchMemOp_v2 (hybrid go)",
        )
    }

    /// Enqueue the wait: until the counter is at least one, then minus one.
    /// Row 0's.
    pub fn enqueue_back(&self, stream: &CudaStream) -> Result<(), GpuError> {
        self.enqueue_back_of(stream, 0)
    }

    /// [`Boundary::enqueue_back`] on row `row`'s counter.
    pub fn enqueue_back_of(&self, stream: &CudaStream, row: usize) -> Result<(), GpuError> {
        self.page_of(row, "Boundary::enqueue_back")?;
        let cnt = self.page.dev_at(Word::Cnt(row).offset());
        mem_batch(
            stream,
            &mut [op_wait_geq(cnt, 1), op_add(cnt, u32::MAX)],
            "cuStreamBatchMemOp_v2 (hybrid wait)",
        )
    }

    /// The device addresses a go of row `row` writes: the row's layer word,
    /// the generation and the region's sequence word.
    pub(super) fn go_words(
        &self,
        row: usize,
        what: &'static str,
    ) -> Result<(sys::CUdeviceptr, sys::CUdeviceptr, sys::CUdeviceptr), GpuError> {
        self.page_of(row, what)?;
        Ok((
            self.page.dev_at(Word::Lyr(row).offset()),
            self.page.dev_at(Word::Gen.offset()),
            self.region.cu_deviceptr() + 4 * self.layout.handoff().seq as u64,
        ))
    }

    /// The device address of row `row`'s counter, which a wait takes back.
    pub(super) fn cnt_word(
        &self,
        row: usize,
        what: &'static str,
    ) -> Result<sys::CUdeviceptr, GpuError> {
        self.page_of(row, what)?;
        Ok(self.page.dev_at(Word::Cnt(row).offset()))
    }

    /// The region's sequence word, which every go adds one to.
    pub(super) fn seq_word(&self) -> &DeviceBuffer<u32> {
        &self.seq
    }

    /// Release every wait still pending in the stream, for good: every
    /// row's counter goes far past anything a step subtracts.
    pub(super) fn release(&self) {
        for row in 0..self.pages.len() {
            word(&self.page, Word::Cnt(row)).store(RELEASE, Ordering::Release);
        }
    }

    /// Row 0's host sum as the page holds it now.
    pub(super) fn hsum_copy(&self, what: &'static str) -> Result<Vec<f32>, GpuError> {
        let off = self.page_of(0, what)?.hsum_off;
        self.page
            .f32_copy(off, self.layout.hsum_len())
            .ok_or(GpuError::state(what, "the sum is outside the page"))
    }
}

// ------------------------------------------------------------------ chains

/// A chain the host tier serves: the one-token step, the two-row pass
/// whose rows run one layer apart (row `r`'s go of layer `l` and the other
/// row's of the layer before can both be in flight), or one row of `m`
/// columns (2..=`DEFER_MAX_COLS`) — `m` consecutive positions whose layer is
/// served by one go, one union call and one wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Step,
    Pair,
    Cols(usize),
}

/// Chains with a capture record of their own: the step, the pair and one per
/// column count up to `DEFER_MAX_COLS` ([`Chain::index`]).
const CHAINS: usize = 2 + DEFER_MAX_COLS;

impl Chain {
    /// The chain's capture record; a `Cols` chain is made only by
    /// [`StepPort::open`], inside `2..=DEFER_MAX_COLS`.
    fn index(self) -> usize {
        match self {
            Chain::Step => 0,
            Chain::Pair => 1,
            Chain::Cols(m) => 1 + m,
        }
    }

    /// Goes of the chain that can have landed past the one being served,
    /// plus one: the rows in flight.
    pub(super) fn in_flight(self) -> usize {
        match self {
            Chain::Step | Chain::Cols(_) => 1,
            Chain::Pair => 2,
        }
    }

    /// Columns one go of the chain carries.
    #[must_use]
    pub fn cols(self) -> usize {
        match self {
            Chain::Step | Chain::Pair => 1,
            Chain::Cols(m) => m,
        }
    }

    /// The chain of a walk of `units` rows of `cols` columns on a page of
    /// `page_cols` columns a row: one row of one column is the step, two the
    /// pair, one row of 2..=`page_cols` columns `Cols`; `None` for any other
    /// point.
    #[must_use]
    pub fn of(units: usize, cols: usize, page_cols: usize) -> Option<Chain> {
        match (units, cols) {
            (1, 1) => Some(Chain::Step),
            (2, 1) => Some(Chain::Pair),
            (1, m) if (2..=page_cols.min(DEFER_MAX_COLS)).contains(&m) => Some(Chain::Cols(m)),
            _ => None,
        }
    }
}

/// What the step port has done since load. Counted by the decode thread
/// alone; nothing here takes a lock. The fields are [`super::HybridStats`]'s
/// of the same names.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StepStats {
    pub(super) served: u64,
    pub(super) go_early: u64,
    pub(super) go_early_first: u64,
    pub(super) host_slots: u64,
    pub(super) host_w2: f64,
    pub(super) leg_ns: u64,
    pub(super) parks_in_service: u64,
    pub(super) straggle_ns: u64,
    pub(super) straggle_max_ns: u64,
    pub(super) overlap_slots: u64,
    pub(super) pair_row1_slots: u64,
    pub(super) host_calls: u64,
    pub(super) cols_served: u64,
    pub(super) cols_cols: u64,
    pub(super) gaps: u64,
}

/// The step port: the boundary and the service state of the go/wait
/// protocol — the sequence served, each chain's captured services, the chain
/// being enqueued, a step refusal not yet named, the pair's overlap record —
/// and its counters. A `Cols` chain's service reads every column of its
/// image and calls the host experts once for all of them
/// ([`StepPort::serve_cols`]); the step's and the pair's read one.
pub struct StepPort {
    pub(crate) boundary: Boundary,
    /// The host's copy of a handoff's activation — one column, allocated at
    /// load.
    x: Tensor2,
    /// A service's host list, in slot order, with room for the boundary's
    /// `n_used` slots made at load; refilled by each service.
    list: Vec<(u32, f32)>,
    /// A `Cols` service's columns (`cols · hidden`), its host lists
    /// (`n_used` a column, each column's first `lens[j]` its list) and the
    /// lists' slices, with room for the page's columns made at load.
    xs: Vec<f32>,
    lists: Vec<(u32, f32)>,
    lens: Vec<usize>,
    slices: Vec<&'static [(u32, f32)]>,
    /// Services done: the sequence number the next handoff carries.
    pub(super) served: u32,
    /// Per [`Chain`], the (layer, row) services its last capture recorded,
    /// in go order — what a replay of it asks the host to serve.
    captured: [Vec<(usize, usize)>; CHAINS],
    /// The chain being enqueued, and whether it is a capture, not an eager
    /// step.
    chain: Chain,
    capturing: bool,
    /// A step service's refusal its step's caller has not yet named.
    pub(super) step_refusal: Option<Refusal>,
    /// Routed slots the last one-column service's handoff sent to each tier
    /// card, by tier: the tiers' hits of that layer, which the host tier
    /// counts.
    pub(super) tier_slots: [u64; MAX_TIERS],
    /// For the row overlap in `stats`: the layer whose host ids row 0 of a
    /// two-row pass served last ([`Chain::Pair`] services only), `None` until
    /// a pair's row 0 is served, and those ids, with room for `n_used` made
    /// at load.
    pair_row0: Option<usize>,
    row0_ids: Vec<u32>,
    pub(super) stats: StepStats,
    /// The route trace, when one is attached: each one-row step's routed
    /// ids, recorded before the service's signal and written after it.
    pub(super) trace: Option<RouteTrace>,
    /// The routed ids each service sees, for the residency rule
    /// ([`super::swap::Tally`]); off without a residency machine.
    pub(super) tally: super::swap::Tally,
    /// The residency machine's staging window, held open while the pool
    /// waits for a go; `None` without a machine.
    pub(super) window: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
    /// Each service's wait for its go, from the host's start of the wait to
    /// the go seen (ns): the card's time between the host's last signal and
    /// this layer's go. A ring of [`GAP_RING`] made at load; `stats.gaps`
    /// counts the waits ever written.
    gaps: Vec<u64>,
}

/// Go waits the ring keeps ([`StepPort::gap_summary`]).
pub const GAP_RING: usize = 4096;

/// The go waits of a span of services ([`StepPort::gap_summary`]), ns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GapSummary {
    pub n: u64,
    pub min_ns: u64,
    /// The lower median.
    pub p50_ns: u64,
    pub max_ns: u64,
}

/// The summary of waits `from..to` of a ring of `ring.len()` that has had
/// `written` waits written in turn, or why not.
fn gap_span(ring: &[u64], written: u64, from: u64, to: u64) -> Result<GapSummary, String> {
    let len = ring.len() as u64;
    if from > to || to > written {
        return Err(format!("services {from}..{to} of {written} written"));
    }
    if written - from > len {
        return Err(format!(
            "services {from}..{to}: the ring of {len} holds the last {len} of {written}"
        ));
    }
    let n = to - from;
    if n == 0 {
        return Ok(GapSummary::default());
    }
    let at = |k: u64| ring[usize::try_from(k % len).unwrap_or(0)];
    let mut out = GapSummary {
        n,
        min_ns: u64::MAX,
        ..GapSummary::default()
    };
    for k in from..to {
        out.min_ns = out.min_ns.min(at(k));
        out.max_ns = out.max_ns.max(at(k));
    }
    // The lower median by counting: the value with at most (n-1)/2 below it
    // and more than (n-1)/2 at or below it.
    let half = (n - 1) / 2;
    for k in from..to {
        let v = at(k);
        let (mut below, mut upto) = (0u64, 0u64);
        for j in from..to {
            below += u64::from(at(j) < v);
            upto += u64::from(at(j) <= v);
        }
        if below <= half && upto > half {
            out.p50_ns = v;
            break;
        }
    }
    Ok(out)
}

impl StepPort {
    /// The port over `boundary`, for chains of up to `layers` hybrid layers.
    /// Load-time only.
    pub fn new(boundary: Boundary, layers: usize) -> StepPort {
        let h = boundary.layout.handoff();
        let cols = boundary.layout.cols();
        StepPort {
            boundary,
            x: Tensor2::zeros(h.hidden, 1),
            list: Vec::with_capacity(h.n_used),
            xs: if cols > 1 {
                vec![0.0; cols * h.hidden]
            } else {
                Vec::new()
            },
            lists: vec![(0, 0.0); if cols > 1 { cols * h.n_used } else { 0 }],
            lens: vec![0; if cols > 1 { cols } else { 0 }],
            slices: Vec::with_capacity(if cols > 1 { cols } else { 0 }),
            served: 0,
            captured: std::array::from_fn(|i| match i {
                0 => Vec::with_capacity(layers),
                1 => Vec::with_capacity(Chain::Pair.in_flight() * layers),
                i if i <= cols + 1 => Vec::with_capacity(layers),
                _ => Vec::new(),
            }),
            chain: Chain::Step,
            capturing: false,
            step_refusal: None,
            tier_slots: [0; MAX_TIERS],
            pair_row0: None,
            row0_ids: Vec::with_capacity(h.n_used),
            stats: StepStats::default(),
            trace: None,
            tally: super::swap::Tally::off(),
            window: None,
            gaps: vec![0; GAP_RING],
        }
    }

    /// The go waits of services `from..to` (`stats.gaps` counts), with no
    /// allocation; refused by name when the ring no longer holds them all
    /// or the span runs past the waits written. An empty span is `n = 0`.
    pub(super) fn gap_summary(&self, from: u64, to: u64) -> Result<GapSummary, GpuError> {
        gap_span(&self.gaps, self.stats.gaps, from, to)
            .map_err(|detail| GpuError::shape("StepPort::gap_summary", detail))
    }

    /// The boundary the chain enqueues its handoffs and waits on.
    #[must_use]
    pub fn boundary(&self) -> &Boundary {
        &self.boundary
    }

    /// Open `chain` on `stream`; a pair needs a boundary of two rows.
    pub(super) fn begin(&mut self, stream: &CudaStream, chain: Chain) -> Result<(), GpuError> {
        let what = "Hybrid::begin_chain";
        if let Chain::Cols(m) = chain
            && Chain::of(1, m, self.boundary.layout.cols()) != Some(chain)
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "a chain of {m} columns on a page of {} columns a row",
                    self.boundary.layout.cols()
                ),
            ));
        }
        if chain.in_flight() > self.boundary.rows() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a {chain:?} chain on a boundary of {} rows",
                    self.boundary.rows()
                ),
            ));
        }
        self.chain = chain;
        self.capturing = capturing(stream)?;
        if self.capturing {
            self.captured[chain.index()].clear();
        }
        Ok(())
    }

    /// Open the chain of a walk of `units` rows of `cols` columns on
    /// `stream` ([`Chain::of`] over the page's columns); any other point is
    /// refused by name.
    pub(super) fn open(
        &mut self,
        stream: &CudaStream,
        units: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let page_cols = self.boundary.layout.cols();
        let chain = Chain::of(units, cols, page_cols).ok_or_else(|| {
            GpuError::shape(
                "Hybrid::begin_chain",
                format!(
                    "{units} rows of {cols} columns: the step port serves one row or {} of one \
                     column, or one row of 2..={} columns (the page's)",
                    Chain::Pair.in_flight(),
                    page_cols.min(DEFER_MAX_COLS)
                ),
            )
        })?;
        self.begin(stream, chain)
    }

    /// Whether the chain being enqueued is a capture; if so, note that
    /// `(layer, row)` is enqueued, which a replay then asks the host for.
    pub(super) fn note(&mut self, layer: usize, row: usize) -> bool {
        if self.capturing {
            self.captured[self.chain.index()].push((layer, row));
        }
        self.capturing
    }

    /// Whether the chain being enqueued is a capture.
    pub(super) fn is_capturing(&self) -> bool {
        self.capturing
    }

    /// The chain being enqueued.
    pub(super) fn chain(&self) -> Chain {
        self.chain
    }

    /// The `i`-th service `chain`'s last capture recorded; `None` past them.
    pub(super) fn captured(&self, chain: Chain, i: usize) -> Option<(usize, usize)> {
        self.captured.get(chain.index())?.get(i).copied()
    }

    /// Back to a fresh port's relation after a refusal's reset: `served` at
    /// the card's generation, no refusal to name, no pair record.
    pub(super) fn rest_at(&mut self, generation: u32) {
        self.served = generation;
        self.step_refusal = None;
        self.pair_row0 = None;
    }

    /// Wait for the go of layer `layer` of row `row` in `chain` and check
    /// it: the go landed in time and no later go is past the rows in
    /// flight, the handoff carries the sequence the host is at, the row's
    /// layer word is `layer`. Returns the row's image and sum offsets, the
    /// service's start and the pool's parks at its wait.
    fn take_go(
        &mut self,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
    ) -> Result<Go, GpuError> {
        let what = SERVE;
        let entered = Instant::now();
        let (image_off, hsum_off) = {
            let p = self.boundary.page_of(row, what)?;
            (p.image_off, p.hsum_off)
        };
        let want = self.served.wrapping_add(1);
        let generation = word(&self.boundary.page, Word::Gen);
        let early = !before(generation.load(Ordering::Acquire), want);
        let parks = threads::pool().stats().worker_parks;
        if let Some(w) = &self.window {
            w.store(1, Ordering::Release);
        }
        let (seen, straggle) = wait_go(generation, want, entered + GO_DEADLINE);
        let gap = nanos(entered.elapsed());
        if let Some(w) = &self.window {
            w.store(0, Ordering::Release);
        }
        self.gaps[(self.stats.gaps % GAP_RING as u64) as usize] = gap;
        self.stats.gaps += 1;
        if early {
            self.stats.go_early += 1;
            self.stats.go_early_first += u64::from(opens_replay);
        } else {
            self.stats.straggle_ns += straggle;
            self.stats.straggle_max_ns = self.stats.straggle_max_ns.max(straggle);
        }
        // The goes of the rows in flight after this one may have landed too;
        // one past them means a wait is missing.
        let ahead = usize::try_from(seen.wrapping_sub(want)).unwrap_or(usize::MAX);
        if before(seen, want) || ahead >= chain.in_flight() {
            let detail = if before(seen, want) {
                format!(
                    "the go of layer {layer} did not land in {GO_DEADLINE:?} (generation {seen}, want {want})"
                )
            } else {
                format!(
                    "generation {seen} is {ahead} past {want} with {} row(s) in flight: the card \
                     ran ahead of the host — a hybrid layer's wait is missing",
                    chain.in_flight()
                )
            };
            return Err(GpuError::protocol(what, detail));
        }
        let t0 = Instant::now();
        let page = &self.boundary.page;
        let h = self.boundary.layout.handoff();
        let words = self.boundary.layout.image_words();
        let seq = payload_word(page, image_off, h.seq, words);
        if seq != Some(self.served) {
            return Err(GpuError::protocol(
                what,
                format!(
                    "the handoff carries sequence {seq:?}, the host is at {}: a stale handoff",
                    self.served
                ),
            ));
        }
        let lyr = word(page, Word::Lyr(row)).load(Ordering::Acquire);
        if usize::try_from(lyr).ok() != Some(layer) {
            return Err(GpuError::protocol(
                what,
                format!("row {row}'s go is layer {lyr}'s, the host serves layer {layer}"),
            ));
        }
        Ok(Go {
            image_off,
            hsum_off,
            t0,
            parks,
        })
    }

    /// The service of `go` is done: one added to the row's counter, the
    /// sequence moved on, the service's time and parks counted.
    fn signal(&mut self, row: usize, go: &Go) {
        word(&self.boundary.page, Word::Cnt(row)).fetch_add(1, Ordering::Release);
        self.served = self.served.wrapping_add(1);
        let s = &mut self.stats;
        s.served += 1;
        s.leg_ns += nanos(go.t0.elapsed());
        s.parks_in_service += threads::pool()
            .stats()
            .worker_parks
            .saturating_sub(go.parks);
    }

    /// Serve layer `layer` of row `row` in `chain` with `experts`, recording
    /// a refusal in `health`: wait for the go, check it, build the host list
    /// from `slots`, run the experts into the row's sum and signal. A `Cols`
    /// chain is [`StepPort::serve_cols`]'s.
    #[allow(
        clippy::too_many_arguments,
        reason = "the tier's three parts the service reads, the go it serves and its chain"
    )]
    pub(super) fn serve_one<H: HostExperts>(
        &mut self,
        experts: &mut H,
        health: &mut Health,
        slots: &SlotMap,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
    ) -> Result<(), GpuError> {
        let what = SERVE;
        if let Chain::Cols(_) = chain {
            return self.serve_cols(experts, health, slots, layer, opens_replay, chain);
        }
        let go = self.take_go(layer, row, opens_replay, chain)?;
        let (image_off, hsum_off) = (go.image_off, go.hsum_off);
        let page = &self.boundary.page;
        let h = self.boundary.layout.handoff();
        let words = self.boundary.layout.image_words();
        if slots.row_offset(layer).is_none() {
            return Err(GpuError::state(
                what,
                "a hybrid layer without a slot map row",
            ));
        }
        // The host's slots, in slot order: every routed id the slot map sends
        // to the host, with its weight — at most the page's `n_used`, the
        // list's room. A card's or the tier's id is theirs.
        self.list.clear();
        let (mut w2_host, mut w2_all) = (0.0f64, 0.0f64);
        let mut tier_slots = [0u64; MAX_TIERS];
        let mut unknown = None;
        for s in 0..h.n_used {
            let (Some(id), Some(wb)) = (
                payload_word(page, image_off, h.ids + s, words),
                payload_word(page, image_off, h.weights + s, words),
            ) else {
                return Err(GpuError::shape(what, "the routing is outside the handoff"));
            };
            let w = f32::from_bits(wb);
            w2_all += f64::from(w) * f64::from(w);
            self.tally.note(layer, row, s, id)?;
            match slots.slot(layer, id) {
                Some(Slot::Host) => {
                    self.list.push((id, w));
                    w2_host += f64::from(w) * f64::from(w);
                }
                Some(Slot::Card(_)) => {}
                Some(Slot::Tier { tier, .. }) => tier_slots[tier] += 1,
                None => unknown = unknown.or(Some((s, id))),
            }
        }
        self.tier_slots = tier_slots;
        if !payload_f32_into(page, image_off, h.x, words, &mut self.x.data) {
            return Err(GpuError::shape(
                what,
                "the activation is outside the handoff",
            ));
        }
        // A handoff the card should already have refused — an id no slot
        // knows, a non-finite activation (the norm that wrote it tests every
        // value it writes) — runs no host expert: its sum is NaN, and the
        // service fails by name; the tier then releases the stream.
        let saw = match unknown {
            Some((s, id)) => Some(unknown_id(s, id, slots.n_expert())),
            None => non_finite(&self.x.data),
        };
        let out = self
            .boundary
            .page
            .f32_mut(hsum_off, h.hidden)
            .ok_or(GpuError::state(what, "the sum is outside the page"))?;
        if let Some(saw) = saw {
            out.fill(f32::NAN);
            return Err(self.refuse(health, layer, format!("row {row}: {saw}")));
        }
        experts.experts_into(layer, &self.x, &self.list, out)?;
        self.stats.host_calls += 1;
        // The page holds the handoff until the signal: the ids are read
        // before it, the finished position is written after it.
        let whole = match self.trace.as_mut() {
            Some(t) => {
                let page = &self.boundary.page;
                t.record(layer, chain, slots, |s| {
                    payload_word(page, image_off, h.ids + s, words)
                })?
            }
            None => false,
        };
        self.signal(row, &go);
        if whole && let Some(t) = self.trace.as_mut() {
            t.write_row()?;
        }
        let s = &mut self.stats;
        s.host_slots += self.list.len() as u64;
        s.host_w2 += if w2_all > 0.0 { w2_host / w2_all } else { 0.0 };
        if chain == Chain::Pair {
            self.count_pair_overlap(layer, row);
        }
        Ok(())
    }

    /// Serve layer `layer` of a `Cols` chain's one row: wait for the go and
    /// check it, build each column's host list in slot order from the
    /// image's routing (`n_used` a column), and run the layer's host experts
    /// for every column in one union call
    /// ([`HostExperts::experts_union_into`]), each column's sum bit for bit
    /// what a one-column service writes for it, into the row's sum; then
    /// signal. A column the card should already have refused runs nothing:
    /// every column's sum is NaN and the service fails by name.
    fn serve_cols<H: HostExperts>(
        &mut self,
        experts: &mut H,
        health: &mut Health,
        slots: &SlotMap,
        layer: usize,
        opens_replay: bool,
        chain: Chain,
    ) -> Result<(), GpuError> {
        let what = SERVE;
        let m = chain.cols();
        if self.trace.is_some() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a {chain:?} service while a route trace is attached: it records one-row steps"
                ),
            ));
        }
        let go = self.take_go(layer, 0, opens_replay, chain)?;
        let page = &self.boundary.page;
        let h = self.boundary.layout.handoff();
        let words = self.boundary.layout.image_words();
        let n = h.n_used;
        if self.lens.len() < m || self.lists.len() < m * n || self.xs.len() < m * h.hidden {
            return Err(GpuError::shape(
                what,
                format!("a service of {m} columns; the port was made for fewer"),
            ));
        }
        if slots.row_offset(layer).is_none() {
            return Err(GpuError::state(
                what,
                "a hybrid layer without a slot map row",
            ));
        }
        let (mut w2_host, mut w2_all, mut host_slots) = (0.0f64, 0.0f64, 0u64);
        let mut refused = None;
        for j in 0..m {
            let list = &mut self.lists[j * n..][..n];
            let mut len = 0usize;
            for s in 0..n {
                let (Some(id), Some(wb)) = (
                    payload_word(page, go.image_off, h.ids + j * n + s, words),
                    payload_word(page, go.image_off, h.weights + j * n + s, words),
                ) else {
                    return Err(GpuError::shape(what, "the routing is outside the handoff"));
                };
                let w = f32::from_bits(wb);
                w2_all += f64::from(w) * f64::from(w);
                self.tally.note(layer, j, s, id)?;
                match slots.slot(layer, id) {
                    Some(Slot::Host) => {
                        list[len] = (id, w);
                        len += 1;
                        w2_host += f64::from(w) * f64::from(w);
                    }
                    Some(Slot::Card(_) | Slot::Tier { .. }) => {}
                    None => refused = refused.or(Some((j, unknown_id(s, id, slots.n_expert())))),
                }
            }
            self.lens[j] = len;
            host_slots += len as u64;
        }
        let xs = &mut self.xs[..m * h.hidden];
        if !payload_f32_into(page, go.image_off, h.x, words, xs) {
            return Err(GpuError::shape(
                what,
                "the activation is outside the handoff",
            ));
        }
        if refused.is_none() {
            refused = (0..m)
                .find_map(|j| non_finite(&xs[j * h.hidden..][..h.hidden]).map(|saw| (j, saw)));
        }
        let out = self
            .boundary
            .page
            .f32_mut(go.hsum_off, m * h.hidden)
            .ok_or(GpuError::state(what, "the sum is outside the page"))?;
        if let Some((j, saw)) = refused {
            out.fill(f32::NAN);
            return Err(self.refuse(health, layer, format!("row 0 column {j} of {m}: {saw}")));
        }
        let x =
            Tensor2View::new(xs, h.hidden, m).map_err(|e| GpuError::shape(what, e.to_string()))?;
        let mut lists = reuse_slices(std::mem::take(&mut self.slices));
        lists.extend(
            self.lens[..m]
                .iter()
                .enumerate()
                .map(|(j, &len)| &self.lists[j * n..][..len]),
        );
        let r = experts.experts_union_into(layer, x, &lists, out);
        self.stats.host_calls += 1;
        self.slices = reuse_slices(lists);
        r?;
        self.signal(0, &go);
        let s = &mut self.stats;
        s.host_slots += host_slots;
        s.host_w2 += if w2_all > 0.0 { w2_host / w2_all } else { 0.0 };
        s.cols_served += 1;
        s.cols_cols += m as u64;
        Ok(())
    }

    /// Record a step service's refusal of layer `layer`, which saw `detail`,
    /// and the error it fails with.
    fn refuse(&mut self, health: &mut Health, layer: usize, detail: String) -> GpuError {
        let what = SERVE;
        let r = Refusal {
            what,
            layer,
            detail,
        };
        let e = GpuError::protocol(
            what,
            format!(
                "host saw undefined input: layer {layer}, {}; the card's fault word is read once \
                 the step drains",
                r.detail
            ),
        );
        self.step_refusal = Some(r.clone());
        health.record_refusal(r, true);
        e
    }

    /// The row overlap of a two-row pass, over the host list the service
    /// just served: row 0's service keeps its host ids, row 1's service of
    /// the same layer — the next one served — counts its ids found among
    /// them. A copy of at most `n_used` ids into room made at load and that
    /// many squared compares, no lock; the one-token step never calls it.
    fn count_pair_overlap(&mut self, layer: usize, row: usize) {
        let list = &self.list;
        if row == 0 {
            self.row0_ids.clear();
            self.row0_ids.extend(list.iter().map(|&(id, _)| id));
            self.pair_row0 = Some(layer);
        } else if self.pair_row0.take_if(|l0| *l0 == layer).is_some() {
            let row0 = &self.row0_ids;
            let overlap = list.iter().filter(|(id, _)| row0.contains(id)).count();
            self.stats.overlap_slots += overlap as u64;
            self.stats.pair_row1_slots += list.len() as u64;
        }
    }
}

/// A go a service took ([`StepPort::take_go`]): the row's image and sum
/// offsets in the page, the service's start and the pool's parks at its
/// wait.
struct Go {
    image_off: usize,
    hsum_off: usize,
    t0: Instant,
    parks: u64,
}

/// `v`'s storage as an empty `Vec` of slices of another lifetime: the
/// in-place collect of an empty iterator keeps the allocation, so a buffer of
/// borrowed lists outlives the borrow it held without a copy.
fn reuse_slices<'a, 'b, T>(mut v: Vec<&'a [T]>) -> Vec<&'b [T]> {
    v.clear();
    v.into_iter()
        .map(|_| -> &'b [T] { unreachable!("the vector was cleared") })
        .collect()
}

/// Wait until the generation word has reached `want` or `deadline` has
/// passed, with every pool thread spinning on it — a pool job, so no worker
/// is parked when the go lands and the dispatch that follows starts inside
/// the spin window this job leaves them in. Returns the word as last read,
/// and the nanoseconds from the calling thread seeing it to the job's end
/// (the calling thread runs the last chunk and alone writes that stamp).
fn wait_go(generation: &AtomicU32, want: u32, deadline: Instant) -> (u32, u64) {
    let pool = threads::pool();
    let caller = pool.threads() - 1;
    let base = Instant::now();
    let seen_at = AtomicU64::new(0);
    pool.for_each_chunk(pool.threads(), |chunk| {
        let mut spins = 0u32;
        while before(generation.load(Ordering::Acquire), want) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(DEADLINE_POLL) && Instant::now() > deadline {
                break;
            }
            std::hint::spin_loop();
        }
        if chunk.start == caller {
            let ns = nanos(base.elapsed());
            seen_at.store(ns, Ordering::Relaxed);
        }
    });
    let end = nanos(base.elapsed());
    (
        generation.load(Ordering::Acquire),
        end.saturating_sub(seen_at.load(Ordering::Relaxed)),
    )
}

#[cfg(test)]
mod tests {
    use super::{Boundary, BoundaryShape, CHAINS, Chain, GapSummary, gap_span};
    use cuda_core::CudaContext;
    use model::ops::DEFER_MAX_COLS;
    use std::sync::Arc;

    /// A span of go waits: least, lower median and most over the span, the
    /// ring's wrap followed; an empty span is zero; a span past the waits
    /// written or older than the ring holds is refused.
    #[test]
    fn a_gap_span_is_its_waits_summary() {
        let mut ring = [0u64; 4];
        let mut written = 0u64;
        for v in [9, 3, 7, 1, 5, 8] {
            ring[(written % 4) as usize] = v;
            written += 1;
        }
        // The ring holds waits 2..6: 7, 1, 5, 8.
        assert_eq!(
            gap_span(&ring, written, 2, 6),
            Ok(GapSummary {
                n: 4,
                min_ns: 1,
                p50_ns: 5,
                max_ns: 8
            })
        );
        assert_eq!(
            gap_span(&ring, written, 3, 6),
            Ok(GapSummary {
                n: 3,
                min_ns: 1,
                p50_ns: 5,
                max_ns: 8
            })
        );
        assert_eq!(gap_span(&ring, written, 6, 6), Ok(GapSummary::default()));
        assert!(
            gap_span(&ring, written, 1, 6).is_err(),
            "wait 1 is overwritten"
        );
        assert!(
            gap_span(&ring, written, 5, 7).is_err(),
            "wait 6 is not written"
        );
    }

    /// A walk's point names its chain: one row of one column the step, two
    /// the pair, one row of 2 up to the page's columns `Cols`, one go a
    /// layer each; anything else has none. Every chain has a capture record
    /// of its own.
    #[test]
    fn a_walk_point_names_its_chain() {
        assert_eq!(Chain::of(1, 1, 1), Some(Chain::Step));
        assert_eq!(Chain::of(2, 1, 4), Some(Chain::Pair));
        assert_eq!(Chain::of(1, 4, 4), Some(Chain::Cols(4)));
        for bad in [
            (1, 5, 4),
            (1, 2, 1),
            (2, 2, 4),
            (0, 1, 1),
            (1, 0, 4),
            (3, 1, 4),
        ] {
            assert_eq!(Chain::of(bad.0, bad.1, bad.2), None, "{bad:?}");
        }
        let mut seen = [false; CHAINS];
        let chains = [Chain::Step, Chain::Pair]
            .into_iter()
            .chain((2..=DEFER_MAX_COLS).map(Chain::Cols));
        for c in chains {
            assert!(!std::mem::replace(&mut seen[c.index()], true), "{c:?}");
            assert_eq!(c.in_flight(), if c == Chain::Pair { 2 } else { 1 });
            assert_eq!(Chain::of(c.in_flight(), c.cols(), DEFER_MAX_COLS), Some(c));
        }
    }

    /// A boundary gives back every context handle its windows took: the
    /// context's count after one is dropped is the count before it was made.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_boundary_gives_back_its_context_handles() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let shape = BoundaryShape {
            hidden: 256,
            n_used: 6,
        };
        let before = Arc::strong_count(&ctx);
        let b = Boundary::with_rows(&ctx, &stream, shape, 2).expect("a boundary");
        let held = Arc::strong_count(&ctx);
        drop(b);
        let after = Arc::strong_count(&ctx);
        eprintln!("boundary context handles: before {before} held {held} after {after}");
        assert!(held > before, "the boundary's buffers hold the context");
        assert_eq!(after, before, "a dropped boundary leaves no context handle");
    }
}
