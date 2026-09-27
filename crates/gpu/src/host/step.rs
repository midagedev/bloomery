//! The step port: the handoff inside the captured chain. The card side is
//! the [`Boundary`] — the handoff region the router's launches write, the
//! host-mapped page ([`PageLayout`]) and the sum the combine reads — and the
//! host side is [`StepPort`], which serves each go in go order: waits for it
//! with the whole pool spinning on the generation word, checks the handoff's
//! sequence number and layer, has the model's host computation write the sum
//! into the page and adds one to the row's counter.

use super::page::{HandoffLayout, PageLayout, Word};
use super::slots::{HOST, SlotMap};
use super::{Health, HostExperts, Refusal, non_finite, unknown_id};
use crate::GpuError;
use crate::graph::{
    MappedHost, capturing, mem_batch, op_add, op_barrier_sys, op_wait_geq, op_write,
};
use crate::tensor::window;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use model::Tensor2;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// What the counter is set to when a service fails: every wait still pending
/// in the stream passes, so a failed step drains instead of hanging.
pub const RELEASE: u32 = 1 << 30;

/// How long a service waits for its go before it gives up and releases the
/// stream.
const GO_DEADLINE: Duration = Duration::from_secs(10);

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
    if from + dst.len() > words {
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
/// at once, served in go order: each row has its own layer word, image and
/// sum in the page; the region, the generation, the counter and the
/// sequence are shared — the region's readers of one row finish on the
/// stream before the next row's norm writes it, and the other three count
/// every go and service in one order.
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
        let what = "Boundary::new";
        let layout = PageLayout::new(rows, 1, shape.hidden, shape.n_used)
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
        // SAFETY: as above, for the activation's `hidden` words at `h.x`, the
        // region's tail.
        let normed = unsafe { window::<f32>(base + 4 * h.x as u64, h.hidden, ctx) };
        // SAFETY: as for the ids, for the sequence's one word at `h.seq`.
        let seq = unsafe { window::<u32>(base + 4 * h.seq as u64, 1, ctx) };
        let pages = (0..rows)
            .map(|r| {
                let (image_off, hsum_off) = layout
                    .image_off(r)
                    .zip(layout.hsum_off(r))
                    .ok_or_else(|| GpuError::shape(what, format!("row {r} of the page")))?;
                // SAFETY: row r's image — `words` words from a 256-aligned
                // offset — and its sum — `hidden` f32 from a 256-aligned
                // offset — lie inside the page (sized by the layout past the
                // last sum), apart from each other and from every other
                // row's; the page moves into the boundary beside the windows
                // and is freed only when the boundary drops.
                let (image, hsum) = unsafe {
                    (
                        window::<u32>(page.dev_at(image_off), words, ctx),
                        window::<f32>(page.dev_at(hsum_off), h.hidden, ctx),
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

    /// The handoff's activation, `hidden` f32: what the norm writes on a
    /// hybrid layer and the host reads.
    #[must_use]
    pub fn normed(&self) -> &DeviceBuffer<f32> {
        &self.normed
    }

    /// The handoff's activation, for the norm to write.
    pub fn normed_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.normed
    }

    /// The host experts' weighted sum, `hidden` f32 in the host-mapped page:
    /// what a combine reads in place after the layer's wait. Row 0's.
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

/// A chain the host tier serves: the one-token step, or the two-row pass
/// whose rows run one layer apart (row `r`'s go of layer `l` and the other
/// row's of the layer before can both be in flight).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Step,
    Pair,
}

impl Chain {
    fn index(self) -> usize {
        match self {
            Chain::Step => 0,
            Chain::Pair => 1,
        }
    }

    /// Goes of the chain that can have landed past the one being served,
    /// plus one: the rows in flight.
    pub(super) fn in_flight(self) -> usize {
        match self {
            Chain::Step => 1,
            Chain::Pair => 2,
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
}

/// The step port: the boundary and the service state of the go/wait
/// protocol — the sequence served, each chain's captured services, the chain
/// being enqueued, a step refusal not yet named, the pair's overlap record —
/// and its counters.
pub struct StepPort {
    pub(crate) boundary: Boundary,
    /// The host's copy of a handoff's activation — one column, allocated at
    /// load.
    x: Tensor2,
    /// A service's host list, in slot order, with room for the boundary's
    /// `n_used` slots made at load; refilled by each service.
    list: Vec<(u32, f32)>,
    /// Services done: the sequence number the next handoff carries.
    pub(super) served: u32,
    /// Per [`Chain`], the (layer, row) services its last capture recorded,
    /// in go order — what a replay of it asks the host to serve.
    captured: [Vec<(usize, usize)>; 2],
    /// The chain being enqueued, and whether it is a capture, not an eager
    /// step.
    chain: Chain,
    capturing: bool,
    /// A step service's refusal its step's caller has not yet named.
    pub(super) step_refusal: Option<Refusal>,
    /// For the row overlap in `stats`: the layer whose host ids row 0 of a
    /// two-row pass served last ([`Chain::Pair`] services only), `None` until
    /// a pair's row 0 is served, and those ids, with room for `n_used` made
    /// at load.
    pair_row0: Option<usize>,
    row0_ids: Vec<u32>,
    pub(super) stats: StepStats,
}

impl StepPort {
    /// The port over `boundary`, for chains of up to `layers` hybrid layers.
    /// Load-time only.
    pub fn new(boundary: Boundary, layers: usize) -> StepPort {
        let h = boundary.layout.handoff();
        StepPort {
            boundary,
            x: Tensor2::zeros(h.hidden, 1),
            list: Vec::with_capacity(h.n_used),
            served: 0,
            captured: [
                Vec::with_capacity(layers),
                Vec::with_capacity(Chain::Pair.in_flight() * layers),
            ],
            chain: Chain::Step,
            capturing: false,
            step_refusal: None,
            pair_row0: None,
            row0_ids: Vec::with_capacity(h.n_used),
            stats: StepStats::default(),
        }
    }

    /// The boundary the chain enqueues its handoffs and waits on.
    #[must_use]
    pub fn boundary(&self) -> &Boundary {
        &self.boundary
    }

    /// Open `chain` on `stream`; a pair needs a boundary of two rows.
    pub(super) fn begin(&mut self, stream: &CudaStream, chain: Chain) -> Result<(), GpuError> {
        let what = "Hybrid::begin_chain";
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
    /// `stream`: the chain whose rows in flight are `units`, of one column
    /// each ([`Chain::in_flight`]); any other point is refused by name.
    pub(super) fn open(
        &mut self,
        stream: &CudaStream,
        units: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let chain = [Chain::Step, Chain::Pair]
            .into_iter()
            .find(|c| c.in_flight() == units)
            .filter(|_| cols == 1)
            .ok_or_else(|| {
                GpuError::shape(
                    "Hybrid::begin_chain",
                    format!(
                        "{units} rows of {cols} columns: the step port serves one row or {}, of \
                         one column",
                        Chain::Pair.in_flight()
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

    /// The chain being enqueued.
    pub(super) fn chain(&self) -> Chain {
        self.chain
    }

    /// The `i`-th service `chain`'s last capture recorded; `None` past them.
    pub(super) fn captured(&self, chain: Chain, i: usize) -> Option<(usize, usize)> {
        self.captured[chain.index()].get(i).copied()
    }

    /// Back to a fresh port's relation after a refusal's reset: `served` at
    /// the card's generation, no refusal to name, no pair record.
    pub(super) fn rest_at(&mut self, generation: u32) {
        self.served = generation;
        self.step_refusal = None;
        self.pair_row0 = None;
    }

    /// Serve layer `layer` of row `row` in `chain` with `experts`, recording
    /// a refusal in `health`: wait for the go, check it, build the host list
    /// from `slots`, run the experts into the row's sum and signal.
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
        let entered = Instant::now();
        let (image_off, hsum_off) = {
            let p = self.boundary.page_of(row, what)?;
            (p.image_off, p.hsum_off)
        };
        let want = self.served.wrapping_add(1);
        let generation = word(&self.boundary.page, Word::Gen);
        let early = !before(generation.load(Ordering::Acquire), want);
        let parks = threads::pool().stats().worker_parks;
        let (seen, straggle) = wait_go(generation, want, entered + GO_DEADLINE);
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
        let map_row = slots.row(layer).ok_or(GpuError::state(
            what,
            "a hybrid layer without a slot map row",
        ))?;
        // The host's slots, in slot order: every routed id the slot map sends
        // to the host, with its weight — at most the page's `n_used`, the
        // list's room.
        self.list.clear();
        let (mut w2_host, mut w2_all) = (0.0f64, 0.0f64);
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
            match usize::try_from(id).ok().and_then(|id| map_row.get(id)) {
                Some(&HOST) => {
                    self.list.push((id, w));
                    w2_host += f64::from(w) * f64::from(w);
                }
                Some(_) => {}
                None => unknown = unknown.or(Some((s, id))),
            }
        }
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
            Some((s, id)) => Some(unknown_id(s, id, map_row.len())),
            None => non_finite(&self.x.data),
        };
        let out = self
            .boundary
            .page
            .f32_mut(hsum_off, h.hidden)
            .ok_or(GpuError::state(what, "the sum is outside the page"))?;
        if let Some(saw) = saw {
            out.fill(f32::NAN);
            let r = Refusal {
                what,
                layer,
                detail: format!("row {row}: {saw}"),
            };
            let e = GpuError::protocol(
                what,
                format!(
                    "host saw undefined input: layer {layer}, {}; the card's fault word is \
                     read once the step drains",
                    r.detail
                ),
            );
            self.step_refusal = Some(r.clone());
            health.record_refusal(r, true);
            return Err(e);
        }
        experts.experts_into(layer, &self.x, &self.list, out)?;
        word(&self.boundary.page, Word::Cnt(row)).fetch_add(1, Ordering::Release);
        self.served = want;
        let s = &mut self.stats;
        s.served += 1;
        s.host_slots += self.list.len() as u64;
        s.host_w2 += if w2_all > 0.0 { w2_host / w2_all } else { 0.0 };
        s.leg_ns += u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        s.parks_in_service += threads::pool().stats().worker_parks.saturating_sub(parks);
        if chain == Chain::Pair {
            self.count_pair_overlap(layer, row);
        }
        Ok(())
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
            let ns = u64::try_from(base.elapsed().as_nanos()).unwrap_or(u64::MAX);
            seen_at.store(ns, Ordering::Relaxed);
        }
    });
    let end = u64::try_from(base.elapsed().as_nanos()).unwrap_or(u64::MAX);
    (
        generation.load(Ordering::Acquire),
        end.saturating_sub(seen_at.load(Ordering::Relaxed)),
    )
}

#[cfg(test)]
mod tests {
    use super::{Boundary, BoundaryShape};
    use cuda_core::CudaContext;
    use std::sync::Arc;

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
