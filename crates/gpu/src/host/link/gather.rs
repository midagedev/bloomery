//! An all-gather of f32 row spans among N ranks over mirrored pages: each
//! rank's card writes its part of a gather into its own page, the link's
//! ends carry the part to every other rank's page, and each card waits for
//! every other part, then adds the parts in rank order, so every rank holds
//! the same bits ([`Gathers::enqueue_sum`]).
//!
//! Every rank's page has one layout ([`Gathers`]): past the page's own bytes,
//! each gather slot holds its span sets, each set one part per rank, then one
//! counter per set and rank; then the link's tail, a line for each set of
//! each slot in each direction of every pair of ranks. A part crosses as a
//! done of the tier protocol does ([`super::pump`]): its counter is, on the
//! writer's page, the generation the writing card adds one to and, on every
//! other page, the counter the reader's end adds one to and the reading card
//! waits for at least one on and takes back, so a captured wait keeps one
//! fixed target on every replay. Every pair of ranks is one link: the lower
//! rank's end holds the stage's side, the higher rank's the tier's.
//!
//! Between two uses of one set of a slot the ranks run another gather. A
//! rank then rewrites a set's part only after its sum of the gather between,
//! which waited for every other rank's part of it, which each rank wrote
//! after its own sum of the gather before, which read the set. So stream
//! order alone keeps every part until each rank has added it, each counter
//! steps by one between two posts, and no wait beyond the gather's own is
//! needed. With one set, a fast rank rewrites the part a late rank has not
//! added yet.

use std::ops::Range;
use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, sys};

use super::pump::{Crossing, Half, Page, PostCols, Reader, Span, TAIL_LINE};
use super::{End, Memory, Pump, Side, drop_settled, link_tail, release_settled};
use crate::GpuError;
use crate::elem::ElemKernels;
use crate::graph::{MappedHost, op_add, op_barrier_sys, op_wait_geq};
use crate::host::step::RELEASE;
use crate::tensor::WindowMut;

/// Bytes a part's stride and the counters align to: one cache line.
const LINE: usize = 64;

/// The shape of one gather slot: each rank's part holds up to `rows` rows of
/// `width` f32 values, row-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherShape {
    pub rows: usize,
    pub width: usize,
}

/// One span set of one gather slot on a mapped page: `dev` is the device
/// address of the page's first byte in the stream's context.
#[derive(Clone, Copy, Debug)]
pub struct GatherSet {
    pub dev: sys::CUdeviceptr,
    pub slot: usize,
    pub set: usize,
}

/// One slot's place on the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    shape: GatherShape,
    /// Byte of the first set's part of rank 0.
    parts: usize,
    /// Bytes from one part to the next, set-major then rank.
    stride: usize,
    /// Byte of the first set's counter of rank 0, one line each.
    counters: usize,
}

impl Slot {
    /// The bytes of `rows` rows of a part.
    fn bytes(&self, rows: usize) -> usize {
        rows * self.shape.width * 4
    }
}

/// The gather slots of one rank's page, laid out alike on every rank's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gathers {
    ranks: usize,
    rank: usize,
    slots: Vec<Slot>,
    /// The byte past the last slot, where the link's tail starts.
    end: usize,
}

impl Gathers {
    /// Span sets per slot.
    pub const SETS: usize = 2;

    /// The page of `rank` of `ranks` with `bytes` of its own before the
    /// gather slots of `shapes`. Fewer than two ranks, a rank past them, an
    /// empty shape or a page past `usize` are refused by name.
    pub fn new(
        bytes: usize,
        ranks: usize,
        rank: usize,
        shapes: &[GatherShape],
    ) -> Result<Gathers, GpuError> {
        const WHAT: &str = "Gathers::new";
        if ranks < 2 || rank >= ranks {
            return Err(GpuError::shape(
                WHAT,
                format!("rank {rank} of {ranks}: a gather joins at least two ranks"),
            ));
        }
        let over = || GpuError::shape(WHAT, "the gather slots overflow the page's byte count");
        let lines = ranks
            .checked_mul(ranks - 1)
            .and_then(|n| (n / 2).checked_mul(shapes.len()))
            .and_then(|n| n.checked_mul(Self::SETS * 2))
            .ok_or_else(over)?;
        let sets = Self::SETS.checked_mul(ranks).ok_or_else(over)?;
        let mut at = bytes.checked_next_multiple_of(LINE).ok_or_else(over)?;
        let mut slots = Vec::with_capacity(shapes.len());
        for &shape in shapes {
            if shape.rows == 0 || shape.width == 0 {
                return Err(GpuError::shape(
                    WHAT,
                    format!("a slot of {shape:?}: want rows and width of at least one"),
                ));
            }
            let part = shape
                .rows
                .checked_mul(shape.width)
                .and_then(|v| v.checked_mul(4))
                .and_then(|b| b.checked_next_multiple_of(LINE))
                .ok_or_else(over)?;
            let parts = at;
            let counters = part
                .checked_mul(sets)
                .and_then(|b| parts.checked_add(b))
                .ok_or_else(over)?;
            at = LINE
                .checked_mul(sets)
                .and_then(|b| b.checked_add(counters))
                .ok_or_else(over)?;
            slots.push(Slot {
                shape,
                parts,
                stride: part,
                counters,
            });
        }
        lines
            .checked_mul(TAIL_LINE)
            .and_then(|b| b.checked_add(at.checked_next_multiple_of(LINE)?))
            .ok_or_else(over)?;
        Ok(Gathers {
            ranks,
            rank,
            slots,
            end: at,
        })
    }

    /// The page's bytes, the link's tail for every pair of ranks included.
    #[must_use]
    pub fn page_bytes(&self) -> usize {
        link_tail(self.end, self.lines()).1
    }

    /// Tail lines of every pair of ranks.
    fn lines(&self) -> usize {
        self.ranks * (self.ranks - 1) / 2 * self.pair_lines()
    }

    /// Tail lines of one pair: each set of each slot, both directions.
    fn pair_lines(&self) -> usize {
        self.slots.len() * Self::SETS * 2
    }

    fn slot(&self, slot: usize) -> &Slot {
        assert!(
            slot < self.slots.len(),
            "gather slot {slot} of {}",
            self.slots.len()
        );
        &self.slots[slot]
    }

    fn checked(&self, slot: usize, set: usize, rank: usize) -> &Slot {
        assert!(
            set < Self::SETS && rank < self.ranks,
            "set {set} of {}, rank {rank} of {}",
            Self::SETS,
            self.ranks
        );
        self.slot(slot)
    }

    /// The bytes of `rank`'s part of `set` of `slot`, at its full rows.
    #[must_use]
    pub fn part(&self, slot: usize, set: usize, rank: usize) -> Range<usize> {
        let s = self.checked(slot, set, rank);
        let start = s.parts + (set * self.ranks + rank) * s.stride;
        start..start + s.bytes(s.shape.rows)
    }

    /// The counter of `rank`'s part of `set` of `slot`: this rank's own
    /// part's generation, another rank's part's delivered count.
    #[must_use]
    pub fn counter(&self, slot: usize, set: usize, rank: usize) -> usize {
        let s = self.checked(slot, set, rank);
        s.counters + (set * self.ranks + rank) * LINE
    }

    /// The other ranks, in rank order.
    fn peers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.ranks).filter(|&p| p != self.rank)
    }

    /// The pair of this rank and `peer`, lower rank first; `peer` is another
    /// rank of the gather.
    fn pair(&self, peer: usize) -> (usize, usize) {
        assert!(
            peer < self.ranks && peer != self.rank,
            "peer {peer} of rank {} of {}",
            self.rank,
            self.ranks
        );
        (self.rank.min(peer), self.rank.max(peer))
    }

    /// The side this rank's end of its link to `peer` holds: the lower
    /// rank's is the stage's.
    #[must_use]
    pub fn side(&self, peer: usize) -> Reader {
        if self.pair(peer).0 == self.rank {
            Reader::Stage
        } else {
            Reader::Tier
        }
    }

    /// The first byte of the tail lines of the link between this rank and
    /// `peer`, the same on both ranks' pages.
    #[must_use]
    pub fn tail(&self, peer: usize) -> usize {
        let (a, b) = self.pair(peer);
        let pair = a * (2 * self.ranks - a - 1) / 2 + (b - a - 1);
        link_tail(self.end, pair * self.pair_lines()).1
    }

    /// What crosses the link between this rank and `peer`, in the order both
    /// ends take: each slot, each set, the lower rank's part then the
    /// higher's. A part's columns are its rows, so an end posts the rows its
    /// [`super::PostCols`] holds.
    #[must_use]
    pub fn crossings(&self, peer: usize) -> Vec<Crossing> {
        let (a, b) = self.pair(peer);
        let mut c = Vec::with_capacity(self.pair_lines());
        for (i, s) in self.slots.iter().enumerate() {
            for set in 0..Self::SETS {
                for (writer, reader) in [(a, Reader::Tier), (b, Reader::Stage)] {
                    c.push(Crossing {
                        counter: self.counter(i, set, writer),
                        reader,
                        spans: vec![Span {
                            bytes: self.part(i, set, writer),
                            cols: s.shape.rows,
                        }],
                        prog: None,
                    });
                }
            }
        }
        c
    }

    /// The stream memory operations that post this rank's part of the set
    /// `at`, once its card has written it: a system-scope barrier, then one
    /// more generation on its counter.
    #[must_use]
    pub fn post(&self, at: GatherSet) -> [sys::CUstreamBatchMemOpParams; 2] {
        let own = self.counter(at.slot, at.set, self.rank) as u64;
        [op_barrier_sys(), op_add(at.dev + own, 1)]
    }

    /// The stream memory operations that wait for every other rank's part of
    /// the set `at`: on each one's counter, a wait for at least one and the
    /// take back of one.
    #[must_use]
    pub fn wait(&self, at: GatherSet) -> Vec<sys::CUstreamBatchMemOpParams> {
        self.peers()
            .flat_map(|p| {
                let counter = at.dev + self.counter(at.slot, at.set, p) as u64;
                [op_wait_geq(counter, 1), op_add(counter, u32::MAX)]
            })
            .collect()
    }

    /// Enqueue `y = ((part_0 + part_1) + part_2) + …` over the first `rows`
    /// rows of every rank's part of the set `at`, in rank order: every
    /// rank adds the same values in the same order, so all hold the same
    /// bits. This order is the gate. More rows than the slot holds, a `y`
    /// shorter than them, or a `y` over the page's parts, counters or tail is
    /// refused by name. Asynchronous, capturable.
    ///
    /// # Safety
    ///
    /// `at.dev` is the device address of the first byte of a page of this
    /// layout, mapped in `stream`'s context, and the page stays allocated
    /// until the enqueued work has run, and for a captured sum until every
    /// launch of the graph has run.
    pub unsafe fn enqueue_sum(
        &self,
        elem: &ElemKernels,
        stream: &CudaStream,
        at: GatherSet,
        rows: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let s = self.checked(at.slot, at.set, 0);
        if rows == 0 || rows > s.shape.rows || y.len() < rows * s.shape.width {
            return Err(GpuError::shape(
                "Gathers::enqueue_sum",
                format!(
                    "{rows} rows of a slot of {:?} into {} values",
                    s.shape,
                    y.len()
                ),
            ));
        }
        let n = rows * s.shape.width;
        let out = y.cu_deviceptr();
        let out_end = out + (n * size_of::<f32>()) as u64;
        let first = at.dev + self.slots[0].parts as u64;
        let end = at.dev + self.page_bytes() as u64;
        if out < end && first < out_end {
            return Err(GpuError::shape(
                "Gathers::enqueue_sum",
                "output overlaps gather parts, counters or link tail",
            ));
        }
        let ctx = stream.context();
        let part = |rank: usize| {
            let start = at.dev + self.part(at.slot, at.set, rank).start as u64;
            // SAFETY: the part's first `n` values lie in the page (this
            // layout's, `n` within its rows), mapped in `stream`'s context and
            // alive for every launch that reads it: the caller's contract.
            unsafe { WindowMut::<f32>::from_raw(start, n, ctx) }
        };
        elem.enqueue_add(stream, &part(0), &part(1), n, y)?;
        for rank in 2..self.ranks {
            // SAFETY: `y` is a live buffer of at least `n` values; the window
            // only names its address for this launch, whose threads each read
            // a value of it before writing that same value.
            let acc = unsafe { WindowMut::<f32>::from_raw(y.cu_deviceptr(), n, ctx) };
            elem.enqueue_add(stream, &acc, &part(rank), n, y)?;
        }
        Ok(())
    }
}

/// One live loopback owns each page until its pump stops and its posts settle.
static JOINED: std::sync::Mutex<std::collections::BTreeSet<usize>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

struct Joined<P>(Vec<Arc<P>>);

impl<P> Joined<P> {
    fn new(pages: &[Arc<P>]) -> Result<Self, GpuError> {
        let mut joined = JOINED
            .lock()
            .expect("the gather page registry is not poisoned");
        for (rank, page) in pages.iter().enumerate() {
            if joined.contains(&(Arc::as_ptr(page) as usize)) {
                return Err(GpuError::shape(
                    "the gather loopback",
                    format!("rank {rank}: page is already joined"),
                ));
            }
        }
        for page in pages {
            joined.insert(Arc::as_ptr(page) as usize);
        }
        Ok(Joined(pages.to_vec()))
    }
}

impl<P> Drop for Joined<P> {
    fn drop(&mut self) {
        let mut joined = JOINED
            .lock()
            .expect("the gather page registry is not poisoned");
        for page in &self.0 {
            assert!(
                joined.remove(&(Arc::as_ptr(page) as usize)),
                "a gather owns its joined page"
            );
        }
    }
}

/// N ranks' pages joined pairwise by the loopback's wire, every end on one
/// pump thread, each end posting the rows `cols` holds.
///
/// The pump stops before the ends settle and drop ([`drop_settled`]).
struct Ring<P: Page + 'static> {
    pumps: Vec<Pump>,
    ends: Vec<Arc<dyn Side<P>>>,
    pages: Joined<P>,
    cols: PostCols,
    layouts: Vec<Gathers>,
    released: std::sync::atomic::AtomicBool,
}

impl<P: Page + 'static> Ring<P> {
    /// Rank `r` of `layouts` on `pages[r]`, a link for every pair of ranks.
    /// Pages of another count than the layouts, a layout that is not rank
    /// `r` of them or whose slots and tail lie unlike rank 0's, and crossings
    /// that do not fit before a page's tail are refused by name.
    fn new(layouts: &[Gathers], pages: &[Arc<P>], cols: &PostCols) -> Result<Ring<P>, GpuError> {
        const WHAT: &str = "the gather loopback";
        if pages.len() != layouts.len() {
            return Err(GpuError::shape(
                WHAT,
                format!("{} pages for {} ranks", pages.len(), layouts.len()),
            ));
        }
        let first = layouts
            .first()
            .ok_or_else(|| GpuError::shape(WHAT, "no ranks"))?;
        for (rank, l) in layouts.iter().enumerate() {
            if l.ranks != layouts.len()
                || l.rank != rank
                || l.end != first.end
                || l.slots != first.slots
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "place {rank}: rank {} of {} unlike rank 0's layout",
                        l.rank, l.ranks
                    ),
                ));
            }
        }
        for (rank, page) in pages.iter().enumerate() {
            if page.bytes() < layouts[rank].page_bytes() {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "rank {rank}: page has {} bytes, needs {}",
                        page.bytes(),
                        layouts[rank].page_bytes()
                    ),
                ));
            }
            if pages[..rank].iter().any(|p| Arc::ptr_eq(p, page)) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("rank {rank}: each rank needs a distinct page"),
                ));
            }
        }
        if first.slots.is_empty() {
            return Err(GpuError::shape(WHAT, "no gather slots"));
        }
        if cols.get() == 0 {
            return Err(GpuError::shape(WHAT, "no rows to post"));
        }
        let joined = Joined::new(pages)?;
        for (rank, (layout, page)) in layouts.iter().zip(pages).enumerate() {
            let counters = (0..layout.slots.len()).flat_map(|slot| {
                (0..Gathers::SETS).flat_map(move |set| {
                    (0..layout.ranks).map(move |peer| layout.counter(slot, set, peer))
                })
            });
            if counters
                .chain((layout.end..layout.page_bytes()).step_by(4))
                .any(|off| page.word(off).load(std::sync::atomic::Ordering::Acquire) != 0)
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!("rank {rank}: gather counters and tail must be zero"),
                ));
            }
        }
        let mut ends: Vec<Arc<dyn Side<P>>> = Vec::new();
        for (a, la) in layouts.iter().enumerate() {
            for b in (0..layouts.len()).filter(|&b| b != a) {
                let half = Half::new(
                    la.crossings(b),
                    la.side(b),
                    la.tail(b),
                    &*pages[a],
                    cols.clone(),
                )
                .map_err(|e| GpuError::shape(WHAT, format!("ranks {a} and {b}: {e}")))?;
                let wire = Memory {
                    from: Arc::clone(&pages[a]),
                    to: Arc::clone(&pages[b]),
                };
                ends.push(Arc::new(End::new(half, wire, Arc::clone(&pages[a]))));
            }
        }
        let pump = Pump::start(ends.clone())
            .map_err(|e| GpuError::shape(WHAT, format!("the pump thread: {e}")))?;
        Ok(Ring {
            pumps: vec![pump],
            ends,
            pages: joined,
            cols: cols.clone(),
            layouts: layouts.to_vec(),
            released: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Refuse a page outside this live loopback or a different row count before a launch.
    fn check(&self, layout: &Gathers, page: &Arc<P>, rows: usize) -> Result<(), GpuError> {
        const WHAT: &str = "the gather loopback";
        let rank = layout.rank;
        if self.layouts.get(rank) != Some(layout) {
            return Err(GpuError::shape(
                WHAT,
                format!("rank {rank}: layout is not joined by this link"),
            ));
        }
        if !self.pages.0.get(rank).is_some_and(|p| Arc::ptr_eq(p, page)) {
            return Err(GpuError::shape(
                WHAT,
                format!("rank {rank}: page is not joined by this link"),
            ));
        }
        if self.released.load(std::sync::atomic::Ordering::Acquire) {
            return Err(GpuError::shape(WHAT, "the link was released"));
        }
        if let Some(why) = self.broken() {
            return Err(GpuError::shape(WHAT, format!("the link is broken: {why}")));
        }
        let posted = self.cols.get();
        if rows == 0
            || layout
                .slots
                .iter()
                .any(|s| posted.min(s.shape.rows) != rows)
        {
            return Err(GpuError::shape(
                WHAT,
                format!("{rows} rows requested, the link posts {posted}"),
            ));
        }
        Ok(())
    }

    /// Why an end stopped carrying, if one did.
    fn broken(&self) -> Option<String> {
        self.ends
            .iter()
            .find_map(|e| e.broken())
            .or_else(|| self.pumps.iter().find_map(Pump::ended))
    }

    /// Release every counter the ends read ([`release_settled`]).
    fn release(&self) -> Result<(), String> {
        self.released
            .store(true, std::sync::atomic::Ordering::Release);
        let ends: Vec<&dyn Side<P>> = self.ends.iter().map(|e| &**e).collect();
        release_settled(&ends, RELEASE)
    }
}

impl<P: Page + 'static> Drop for Ring<P> {
    fn drop(&mut self) {
        self.pumps.clear();
        drop_settled(&self.ends);
    }
}

/// The gathers of N ranks in one process, each rank's page a mapped host
/// page of its own card context, the parts carried by the loopback: what a
/// gate runs every rank of a tensor-parallel load on, the wire a copy in
/// process, as the tier link's loopback carries a tier page's.
pub struct GatherLoop(Ring<MappedHost>);

impl GatherLoop {
    /// Rank `r` of `layouts` (one [`Gathers`] a rank, alike) on `pages[r]`,
    /// every end posting the rows `cols` holds. Pages of another count than
    /// the layouts, layouts that are not ranks 0..N of N laid out alike, and
    /// duplicate, undersized or already joined pages, nonzero counters or tails,
    /// empty slots, no rows to post, or a layout whose crossings do not fit
    /// before its tail are refused by name.
    pub fn new(
        layouts: &[Gathers],
        pages: &[Arc<MappedHost>],
        cols: &PostCols,
    ) -> Result<GatherLoop, GpuError> {
        Ring::new(layouts, pages, cols).map(GatherLoop)
    }

    /// Check that `layout` and `page` belong to this live link, posting `rows` rows.
    pub fn check(
        &self,
        layout: &Gathers,
        page: &Arc<MappedHost>,
        rows: usize,
    ) -> Result<(), GpuError> {
        self.0.check(layout, page, rows)
    }

    /// Why the loopback stopped carrying, if it did.
    #[must_use]
    pub fn broken(&self) -> Option<String> {
        self.0.broken()
    }

    /// Release every counter a rank's card waits on, so that a card stuck on
    /// a part that will not come runs on: for a run that failed with a rank
    /// waiting. `Err` names the ends that could not settle.
    pub fn release(&self) -> Result<(), GpuError> {
        self.0
            .release()
            .map_err(|detail| GpuError::shape("GatherLoop::release", detail))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    use super::super::link_tail;
    use super::super::pump::tests::Mem;
    use super::super::pump::{Page, PostCols};
    use super::{GatherSet, GatherShape, Gathers, Ring};
    use crate::Gpu;
    use crate::graph::{MappedHost, mem_batch};
    use crate::tensor::WindowMut;
    use cuda_core::DeviceBuffer;

    /// Value `i` of `rank`'s part of gather `g`: of either sign and of a
    /// scale that grows a thousandfold a rank, so a sum of the parts in
    /// another order rounds differently.
    fn value(g: usize, rank: usize, i: usize) -> f32 {
        let k = (g * 31 + i * 7 + rank * 13) % 97;
        let sign = if (g + i + rank).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let scale = 1e3f32.powi(i32::try_from(rank).expect("a test's rank") - 1);
        sign * (1.0 + k as f32 / 97.0) * scale
    }

    /// The rank counts every test runs: the gather assumes none.
    const RANKS: [usize; 3] = [2, 3, 4];

    /// The parts of gather `g` added in rank order, value `i`.
    fn rank_sum(g: usize, ranks: usize, i: usize) -> f32 {
        (1..ranks).fold(value(g, 0, i), |acc, r| acc + value(g, r, i))
    }

    const SHAPE: GatherShape = GatherShape { rows: 4, width: 24 };
    /// Rows a gather carries: fewer than the slot holds.
    const ROWS: usize = 3;

    fn layouts(ranks: usize) -> Vec<Gathers> {
        (0..ranks)
            .map(|r| Gathers::new(40, ranks, r, &[SHAPE]).expect("a gather page"))
            .collect()
    }

    fn mem_ring(ranks: usize) -> (Vec<Gathers>, Vec<Arc<Mem>>, Ring<Mem>) {
        let layouts = layouts(ranks);
        let words = layouts[0].page_bytes() / 4;
        let pages: Vec<Arc<Mem>> = (0..ranks)
            .map(|_| Arc::new(Mem((0..words).map(|_| AtomicU32::new(0)).collect())))
            .collect();
        let cols = PostCols::default();
        cols.set(ROWS);
        let ring = Ring::new(&layouts, &pages, &cols).expect("the loopback");
        (layouts, pages, ring)
    }

    // The loopback refuses by name, before any pump starts and without a panic, pages of another
    // count than the ranks' layouts, and layouts that are not ranks 0..N of N laid out alike:
    // one-row slots of widths 1 and 2 share a part stride, counters and tail, and still do not
    // join.
    #[test]
    fn a_loopback_of_other_pages_than_its_ranks_is_refused() {
        let (layouts, pages, ring) = mem_ring(2);
        drop(ring);
        let width = |rank, width| {
            Gathers::new(40, 2, rank, &[GatherShape { rows: 1, width }]).expect("a gather page")
        };
        let (narrow, wide) = (width(0, 1), width(1, 2));
        assert_eq!(narrow.page_bytes(), wide.page_bytes());
        let three = Gathers::new(40, 3, 1, &[SHAPE]).expect("a gather page");
        let swapped = vec![layouts[1].clone(), layouts[0].clone()];
        let mut accepted = Vec::new();
        for (what, bad, n, says) in [
            ("one page", layouts.clone(), 1, "1 pages for 2 ranks"),
            ("no ranks", Vec::new(), 0, "no ranks"),
            (
                "widths 1 and 2",
                vec![narrow, wide],
                2,
                "place 1: rank 1 of 2 unlike",
            ),
            ("ranks swapped", swapped, 2, "place 0: rank 1 of 2 unlike"),
            (
                "ranks of 2 and 3",
                vec![layouts[0].clone(), three],
                2,
                "place 1: rank 1 of 3 unlike",
            ),
        ] {
            let got = std::panic::catch_unwind(|| {
                Ring::new(&bad, &pages[..n], &PostCols::default()).map(|_| ())
            });
            if !got
                .as_ref()
                .is_ok_and(|r| r.as_ref().is_err_and(|e| e.to_string().contains(says)))
            {
                accepted.push((what, got.map(|r| r.map_err(|e| e.to_string()))));
            }
        }
        assert!(accepted.is_empty(), "not refused by name: {accepted:?}");
    }

    #[test]
    fn a_launch_needs_its_live_link_and_rows() {
        let (layouts, pages, ring) = mem_ring(2);
        assert!(ring.check(&layouts[0], &pages[0], ROWS).is_ok());
        assert!(ring.check(&layouts[1], &pages[1], ROWS).is_ok());
        let mut other = layouts[0].clone();
        other.ranks = 4;
        for (layout, page, rows, want) in [
            (&layouts[0], &pages[1], ROWS, "page is not joined"),
            (&other, &pages[0], ROWS, "layout is not joined"),
            (&layouts[0], &pages[0], 0, "rows requested"),
            (&layouts[0], &pages[0], SHAPE.rows + 1, "rows requested"),
        ] {
            let e = ring.check(layout, page, rows).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        }
        assert!(
            Ring::new(&layouts, &pages, &ring.cols)
                .err()
                .unwrap()
                .to_string()
                .contains("already joined")
        );
        ring.cols.set(0);
        assert!(
            ring.check(&layouts[0], &pages[0], 1)
                .unwrap_err()
                .to_string()
                .contains("link posts 0")
        );
        assert!(
            Ring::new(&layouts, &pages, &ring.cols)
                .err()
                .unwrap()
                .to_string()
                .contains("no rows")
        );
        ring.cols.set(SHAPE.rows);
        ring.release().expect("release the idle loopback");
        assert!(
            ring.check(&layouts[0], &pages[0], SHAPE.rows)
                .unwrap_err()
                .to_string()
                .contains("link was released")
        );
        let short = Arc::new(Mem(vec![AtomicU32::new(0)]));
        let short_pages = [short, Arc::clone(&pages[1])];
        assert!(
            Ring::new(&layouts, &short_pages, &PostCols::default())
                .err()
                .unwrap()
                .to_string()
                .contains("page has")
        );
        let duplicate = [Arc::clone(&pages[0]), Arc::clone(&pages[0])];
        assert!(
            Ring::new(&layouts, &duplicate, &PostCols::default())
                .err()
                .unwrap()
                .to_string()
                .contains("distinct page")
        );
        drop(ring);
        assert!(
            Ring::new(&layouts, &pages, &PostCols::default())
                .err()
                .unwrap()
                .to_string()
                .contains("must be zero")
        );
        for page in &pages {
            for word in &page.0 {
                word.store(0, Ordering::Relaxed);
            }
        }
        let clean =
            Ring::new(&layouts, &pages, &PostCols::default()).expect("fresh pages can join again");
        pages[0]
            .word(layouts[0].counter(0, 0, 0))
            .store(2, Ordering::Release);
        let until = Instant::now() + Duration::from_secs(3);
        while clean.broken().is_none() && Instant::now() < until {
            std::thread::yield_now();
        }
        assert!(
            clean
                .check(&layouts[0], &pages[0], SHAPE.rows)
                .unwrap_err()
                .to_string()
                .contains("link is broken")
        );
        drop(clean);
        let empty = (0..2)
            .map(|r| Gathers::new(40, 2, r, &[]).unwrap())
            .collect::<Vec<_>>();
        assert!(
            Ring::new(&empty, &pages, &PostCols::default())
                .err()
                .unwrap()
                .to_string()
                .contains("no gather slots")
        );
    }

    /// Wait until the counter at `off` holds at least one, then take one
    /// back unless `peek`.
    fn wait(page: &Mem, off: usize, peek: bool, until: Instant) -> Result<(), String> {
        while page.word(off).load(Ordering::Acquire) < 1 {
            if Instant::now() > until {
                return Err(format!("the counter at byte {off} never came"));
            }
            std::thread::yield_now();
        }
        if !peek {
            page.word(off).fetch_sub(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// A rank's card played by the host over `gathers` gathers, gather g in
    /// set g mod 2: write the part, post it, wait for every other part, add
    /// all in rank order and count the values that are not the host's rank-
    /// order sum. A `late` card, before it adds gather g, waits until every
    /// other rank's part of gather g + 1 has landed on its page.
    fn card(l: &Gathers, page: &Mem, gathers: usize, late: bool) -> Result<usize, String> {
        let until = Instant::now() + Duration::from_secs(20);
        let n = ROWS * SHAPE.width;
        let mut wrong = 0;
        for g in 0..gathers {
            let set = g % Gathers::SETS;
            let own = l.part(0, set, l.rank).start;
            for i in 0..n {
                page.word(own + 4 * i)
                    .store(value(g, l.rank, i).to_bits(), Ordering::Relaxed);
            }
            page.word(l.counter(0, set, l.rank))
                .fetch_add(1, Ordering::Release);
            for p in l.peers() {
                wait(page, l.counter(0, set, p), false, until)?;
            }
            if late && g + 1 < gathers {
                for p in l.peers() {
                    wait(page, l.counter(0, (g + 1) % Gathers::SETS, p), true, until)?;
                }
            }
            for i in 0..n {
                let part = |r: usize| {
                    f32::from_bits(
                        page.word(l.part(0, set, r).start + 4 * i)
                            .load(Ordering::Relaxed),
                    )
                };
                let sum = (1..l.ranks).fold(part(0), |acc, r| acc + part(r));
                wrong += usize::from(sum.to_bits() != rank_sum(g, l.ranks, i).to_bits());
            }
        }
        Ok(wrong)
    }

    /// Every rank's card on a thread of its own over the loopback; the last
    /// rank's card is `late` when asked. Each card's count of wrong values.
    fn run(ranks: usize, gathers: usize, late: bool) -> Vec<Result<usize, String>> {
        let (layouts, pages, ring) = mem_ring(ranks);
        let got: Vec<Result<usize, String>> = std::thread::scope(|s| {
            let cards: Vec<_> = (0..ranks)
                .map(|r| {
                    let (l, page) = (&layouts[r], &pages[r]);
                    s.spawn(move || card(l, page, gathers, late && r + 1 == ranks))
                })
                .collect();
            cards
                .into_iter()
                .map(|c| c.join().expect("a card thread"))
                .collect()
        });
        if let Some(b) = ring.broken() {
            ring.release().expect("the loopback's ends settle");
            panic!("the loopback broke: {b}");
        }
        got
    }

    // Two, three and four ranks, every card on its own thread and every end
    // on one pump: over many gathers, each rank's rank-order sum of what its
    // page holds is the host's rank-order sum of the parts, bit for bit. Past
    // two ranks the rank-order sum is not the reverse-order one, so the order
    // is pinned.
    #[test]
    fn every_rank_adds_every_part_in_rank_order() {
        for ranks in RANKS.into_iter().filter(|&n| n > 2) {
            let reversed = (0..64).any(|i| {
                let rev = (0..ranks - 1)
                    .rev()
                    .fold(value(0, ranks - 1, i), |acc, r| acc + value(0, r, i));
                rev.to_bits() != rank_sum(0, ranks, i).to_bits()
            });
            assert!(
                reversed,
                "{ranks} ranks: the parts must round differently in another order"
            );
        }
        for ranks in RANKS {
            let got = run(ranks, 400, false);
            assert!(
                got.iter().all(|c| *c == Ok(0)),
                "{ranks} ranks: wrong values per rank {got:?}"
            );
        }
    }

    // A late rank adds gather g only after every other rank's part of gather
    // g + 1 has landed on its page: the parts it adds are still gather g's,
    // since gather g + 1 lands in the other set.
    #[test]
    fn a_late_rank_adds_the_parts_of_its_own_gather() {
        for ranks in RANKS {
            let got = run(ranks, 64, true);
            assert!(
                got.iter().all(|c| *c == Ok(0)),
                "{ranks} ranks, the last one late: wrong values per rank {got:?}"
            );
        }
    }

    // Two, three and four ranks on one card, each a `Gpu` of its own with its page
    // in the card's context, one pump thread carrying every part: each
    // rank's captured chain of four gathers writes its part, posts it, waits
    // for every other part and adds them all. After many replays of every
    // rank's chain, each rank's sum of each gather is the host's rank-order
    // sum of the parts, bit for bit. The ranks' streams wait on one another,
    // so beside another test's streams a shared hardware queue can hold them
    // all: `gate-gpu-lib` runs it alone (`hw_held_`).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_every_rank_adds_the_parts_in_rank_order_on_the_card() {
        const GATHERS: usize = 4;
        const REPLAYS: usize = 200;
        let n = ROWS * SHAPE.width;
        for ranks in RANKS {
            let layouts = layouts(ranks);
            let gpus: Vec<Gpu> = (0..ranks)
                .map(|_| Gpu::new().expect("CUDA device 0"))
                .collect();
            let pages: Vec<Arc<MappedHost>> = gpus
                .iter()
                .map(|g| {
                    let bytes = layouts[0].page_bytes();
                    Arc::new(
                        MappedHost::new(g.context(), bytes, "gather test page").expect("a page"),
                    )
                })
                .collect();
            let cols = PostCols::default();
            cols.set(ROWS);
            let ring = Ring::new(&layouts, &pages, &cols).expect("the loopback");
            let dev = pages[0].dev_at(0);
            let over = dev + layouts[0].part(0, 0, ranks - 1).start as u64;
            // SAFETY: the window names rank 0's own page, which outlives it,
            // and the refused sum enqueues nothing through it.
            let mut over = unsafe { WindowMut::<f32>::from_raw(over, n, gpus[0].context()) };
            // SAFETY: `dev` is rank 0's page of this layout, mapped in this
            // context; the sum is refused before any launch.
            let refused = unsafe {
                layouts[0].enqueue_sum(
                    gpus[0].elem(),
                    gpus[0].stream(),
                    GatherSet {
                        dev,
                        slot: 0,
                        set: 0,
                    },
                    ROWS,
                    &mut over,
                )
            };
            assert!(
                refused.is_err(),
                "{ranks} ranks: a sum into a part is refused"
            );
            drop(over);
            let buf =
                |g: &Gpu, v: &[f32]| DeviceBuffer::from_host(g.stream(), v).expect("a buffer");
            let zero: Vec<_> = gpus.iter().map(|g| buf(g, &vec![0.0; n])).collect();
            let parts: Vec<Vec<_>> = (0..ranks)
                .map(|r| {
                    (0..GATHERS)
                        .map(|g| {
                            let v: Vec<f32> = (0..n).map(|i| value(g, r, i)).collect();
                            buf(&gpus[r], &v)
                        })
                        .collect()
                })
                .collect();
            let mut sums: Vec<Vec<_>> = gpus
                .iter()
                .map(|g| (0..GATHERS).map(|_| buf(g, &vec![f32::NAN; n])).collect())
                .collect();
            let mut graphs = Vec::with_capacity(ranks);
            for (r, (gpu, l)) in gpus.iter().zip(&layouts).enumerate() {
                let dev = pages[r].dev_at(0);
                let graph = gpu.capture(|s| {
                    for g in 0..GATHERS {
                        let set = g % Gathers::SETS;
                        let own = dev + l.part(0, set, r).start as u64;
                        // SAFETY: the part's `n` values lie in rank r's page,
                        // declared before the graph, so dropped after it.
                        let mut part = unsafe { WindowMut::<f32>::from_raw(own, n, gpu.context()) };
                        gpu.elem()
                            .enqueue_add(s, &parts[r][g], &zero[r], n, &mut part)?;
                        let at = GatherSet { dev, slot: 0, set };
                        let mut ops = l.post(at).to_vec();
                        ops.extend(l.wait(at));
                        // SAFETY: every counter lies in rank r's page, which
                        // outlives the graph.
                        unsafe { mem_batch(s, &mut ops, "the gather test's post and wait")? };
                        // SAFETY: `dev` is rank r's page of this layout, mapped
                        // in this context and outliving the graph.
                        unsafe { l.enqueue_sum(gpu.elem(), s, at, ROWS, &mut sums[r][g])? };
                    }
                    Ok(())
                });
                graphs.push(graph.expect("a gather chain graph"));
            }
            for _ in 0..REPLAYS {
                for (graph, gpu) in graphs.iter().zip(&gpus) {
                    graph.launch(gpu.stream()).expect("a replay");
                }
            }
            let ends: Vec<_> = gpus
                .iter()
                .map(|g| {
                    let e = g.context().new_event(None).expect("an event");
                    e.record(g.stream()).expect("recorded");
                    e
                })
                .collect();
            let until = Instant::now() + Duration::from_secs(60);
            while !ends.iter().all(|e| e.query().expect("an event query")) {
                if Instant::now() > until || ring.broken().is_some() {
                    ring.release().expect("the loopback's ends settle");
                    for g in gpus.iter() {
                        g.stream().synchronize().expect("drained");
                    }
                    panic!("{ranks} ranks: the chains stalled: {:?}", ring.broken());
                }
                std::thread::yield_now();
            }
            for (r, (gpu, sums)) in gpus.iter().zip(&sums).enumerate() {
                for (g, sum) in sums.iter().enumerate() {
                    let got = sum.to_host_vec(gpu.stream()).expect("the sum");
                    let bad = (0..n).find(|&i| got[i].to_bits() != rank_sum(g, ranks, i).to_bits());
                    assert_eq!(bad, None, "{ranks} ranks: rank {r}'s gather {g}");
                }
            }
        }
    }

    // No gather slot leaves the page's own bytes and its tail where the link
    // puts them. Every rank lays out the same parts, counters and tail lines,
    // and both ends of a link take the same crossings from opposite sides.
    // A gather of fewer than two ranks, an empty slot, a rank past the ranks
    // and a slot or rank count whose page passes `usize` are refused.
    #[test]
    fn every_rank_lays_out_the_same_page() {
        for bytes in [0, 4, 64, 1000] {
            let g = Gathers::new(bytes, 2, 1, &[]).expect("no slot");
            assert_eq!(g.page_bytes(), link_tail(bytes, 0).1);
            assert_eq!(g.tail(0), link_tail(bytes, 0).0);
            assert!(g.crossings(0).is_empty());
        }
        let shapes = [SHAPE, GatherShape { rows: 2, width: 2 }];
        for n in RANKS {
            let ls: Vec<Gathers> = (0..n)
                .map(|r| Gathers::new(40, n, r, &shapes).expect("two slots"))
                .collect();
            for l in &ls {
                assert_eq!(l.page_bytes(), ls[0].page_bytes());
                for (slot, set, r) in [(0, 0, 0), (0, 1, n - 1), (1, 0, 1), (1, 1, n - 1)] {
                    assert_eq!(l.part(slot, set, r), ls[0].part(slot, set, r));
                    assert_eq!(l.counter(slot, set, r), ls[0].counter(slot, set, r));
                }
            }
            // A part of SHAPE fills whole lines.
            assert_eq!(ls[0].part(0, 0, 0), 64..64 + 4 * 24 * 4);
            let last = 64 + (2 * n - 1) * 384;
            assert_eq!(ls[0].part(0, 1, n - 1), last..last + 384);
            assert_eq!(ls[0].counter(0, 0, 0), 64 + 2 * n * 384);
            assert_eq!(ls[0].part(1, 0, 0).start, 64 + 2 * n * (384 + 64));
            let pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .collect();
            for &(a, b) in &pairs {
                assert_eq!(ls[a].tail(b), ls[b].tail(a));
                assert_eq!(ls[a].crossings(b), ls[b].crossings(a));
                assert_ne!(ls[a].side(b), ls[b].side(a));
            }
            // Each pair holds 2 slots x 2 sets x 2 directions of tail lines.
            let tails: Vec<usize> = pairs.iter().map(|&(a, b)| ls[a].tail(b)).collect();
            assert!(tails.windows(2).all(|w| w[1] - w[0] == 8 * 16));
            assert_eq!(ls[0].page_bytes(), tails[tails.len() - 1] + 8 * 16);
        }
        assert!(Gathers::new(0, 1, 0, &shapes).is_err());
        assert!(Gathers::new(0, 2, 2, &shapes).is_err());
        assert!(Gathers::new(0, 2, 0, &[GatherShape { rows: 0, width: 1 }]).is_err());
        assert!(
            Gathers::new(
                0,
                2,
                0,
                &[GatherShape {
                    rows: usize::MAX,
                    width: 2
                }]
            )
            .is_err()
        );
        let past_usize = [
            1usize << (usize::BITS - 1),
            (1usize << (usize::BITS / 2)) + 1,
        ];
        let taken: Vec<(usize, usize)> = [0, 64]
            .into_iter()
            .flat_map(|bytes| past_usize.map(|ranks| (bytes, ranks)))
            .filter(|&(bytes, ranks)| {
                Gathers::new(bytes, ranks, 0, &[GatherShape { rows: 1, width: 1 }]).is_ok()
            })
            .collect();
        assert!(
            taken.is_empty(),
            "pages past usize taken, (bytes, ranks): {taken:?}"
        );
    }
}
