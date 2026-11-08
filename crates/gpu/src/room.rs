//! Device buffers whose address range is reserved whole and whose pages are
//! backed only as far as the rows in use reach, through the driver's virtual
//! memory management (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`). A
//! buffer keeps its length, its layout and its address, so every launch and
//! every capture over it stays as it is; the card holds only the pages the
//! rows a sequence reached lie on.
//!
//! A [`Room`] backs a page with its own memory, zeroed on its stream, the
//! first time a fit reaches it ([`Room::fit`]), and at a sequence's end gives
//! back every page no row of that sequence reached ([`Room::settle`]). A page
//! with no memory of its own maps the device's one spare page ([`spare`]):
//! the driver checks a copy's whole extent, the bytes between its rows too
//! (a 2-D copy of a store's planes), so no page of the range is left
//! unmapped. A load of `ctx` positions a slot so holds the rows its requests
//! use, rounded up to the device's allocation granularity, and a slot that
//! serves requests of one length takes no memory after its first. The plan
//! reserves every buffer's page rounding on every resident slot and the
//! device's spare page before choosing experts or context.
//!
//! Two rooms of one layout (two sequences' stores of one load) share the
//! pages of a common prefix ([`Room::share`]): a page that holds whole rows
//! of the prefix alone maps the other room's memory, and the page where the
//! prefix ends gets memory of its own with the other room's bytes copied in.
//! A shared page is written by neither: a writer first gives every shared
//! page its write reaches memory of its own holding the same bytes
//! ([`Room::own_from`]), and a sequence's end gives its shared pages back
//! ([`Room::settle`]). A page's memory is freed when the last room that maps
//! it lets it go.
//!
//! Every writer calls `fit` and `own_from` first, outside capture, for all
//! positions it can write. An unbacked page aliases the spare: a missed fit
//! can corrupt another room without a device fault, and a missed `own_from`
//! another sequence's rows.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use cuda_core::sys::CUdevice;
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::GpuError;

const WHAT: &str = "room";

/// Rows of one buffer in a room: `rows` rows of `row` bytes from byte `at`
/// of the buffer, one row for every `per` positions (a pooled key's row
/// covers `per` positions, every other row one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) at: usize,
    pub(crate) row: usize,
    pub(crate) per: usize,
    pub(crate) rows: usize,
}

impl Run {
    /// The bytes `[at, end)` the rows of the first `positions` positions take.
    fn span(&self, positions: usize) -> (usize, usize) {
        let rows = positions.div_ceil(self.per).min(self.rows);
        (self.at, self.at + rows * self.row)
    }

    /// The byte of the first row a write from position `pos` on reaches:
    /// the row that holds `pos` (a pooled row a later position completes is
    /// written again).
    fn written_from(&self, pos: usize) -> usize {
        self.at + (pos / self.per).min(self.rows) * self.row
    }
}

/// What a share moved ([`Room::share`]): the bytes of the pages that map
/// the other room's memory, and the bytes copied into pages of their own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedPages {
    pub mapped: usize,
    pub copied: usize,
}

impl std::ops::Add for SharedPages {
    type Output = SharedPages;
    fn add(self, o: SharedPages) -> SharedPages {
        SharedPages {
            mapped: self.mapped + o.mapped,
            copied: self.copied + o.copied,
        }
    }
}

/// A page of a room's range: its mapping, and the memory it maps other
/// than the spare (its own, or another room's too after a share). The
/// mapping drops first (unmapped), then this room's hold on the memory.
struct Page {
    _map: Mapping,
    own: Option<Arc<PhysicalAllocation>>,
}

/// The page every page of `device`'s rooms with no memory of its own maps,
/// and whether it was made just now (its bytes not set yet). One a device
/// while any room holds it. Its bytes are no row's: nothing reads or
/// writes them but an access past the rows in use.
fn spare(device: CUdevice, page: usize) -> Result<(Arc<PhysicalAllocation>, bool), GpuError> {
    static SPARES: Mutex<Vec<(CUdevice, Weak<PhysicalAllocation>)>> = Mutex::new(Vec::new());
    let mut spares = SPARES.lock().unwrap_or_else(PoisonError::into_inner);
    spares.retain(|(_, s)| s.strong_count() > 0);
    if let Some(s) = spares
        .iter()
        .find(|(d, _)| *d == device)
        .and_then(|(_, s)| s.upgrade())
    {
        return Ok((s, false));
    }
    let s = Arc::new(PhysicalAllocation::new(device, page).map_err(driver("cuMemCreate"))?);
    spares.push((device, Arc::downgrade(&s)));
    Ok((s, true))
}

fn driver(op: &'static str) -> impl Fn(DriverError) -> GpuError {
    move |source| GpuError::Driver {
        op: Some(op),
        source,
    }
}

/// The reserved range of `N` buffers of one length, each at a page boundary,
/// and its pages. The pages drop before the reservation (field order), as the
/// driver wants every mapping gone before its range is freed.
pub(crate) struct Room {
    /// The store this room backs, for a fit's named refusal.
    what: &'static str,
    /// Every page of the range; none only after a remap failed.
    pages: Vec<Option<Page>>,
    /// Every buffer's runs, at their offsets in the range.
    runs: Vec<Run>,
    /// The positions whose rows lie on pages with memory of their own: a
    /// fit up to it backs nothing new.
    fitted: usize,
    /// The most positions a fit asked since the last settle.
    reach: usize,
    /// The pages with memory of their own.
    mapped: usize,
    /// Whether a page may map memory another room maps too: set by a share
    /// on both rooms (the one it reads from through `&`), cleared once a
    /// scan finds none, so a sequence that never shared scans no page.
    lent: AtomicBool,
    page: usize,
    spare: Arc<PhysicalAllocation>,
    stream: Arc<CudaStream>,
    reserve: VirtualReservation,
}

impl Room {
    /// The pages any run's rows of the first `positions` positions lie on.
    fn wanted(&self, positions: usize) -> Vec<bool> {
        let mut want = vec![false; self.pages.len()];
        for r in &self.runs {
            let (at, end) = r.span(positions);
            if end > at {
                want[at / self.page..=(end - 1) / self.page].fill(true);
            }
        }
        want
    }

    /// Page `i` mapped read-write for the room's device to `own`, or to the
    /// spare without it, whatever mapped it before unmapped first.
    fn back(&mut self, i: usize, own: Option<Arc<PhysicalAllocation>>) -> Result<(), GpuError> {
        if self.pages[i].take().is_some_and(|p| p.own.is_some()) {
            self.mapped -= 1;
        }
        let va = self.va(i);
        let map = Mapping::new(va, self.page, own.as_deref().unwrap_or(&self.spare), 0)
            .map_err(driver("cuMemMap"))?;
        vmm::set_access(va, self.page, &[self.stream.context().cu_device()])
            .map_err(driver("cuMemSetAccess"))?;
        self.mapped += usize::from(own.is_some());
        self.pages[i] = Some(Page { _map: map, own });
        Ok(())
    }

    /// Page `i`'s address.
    fn va(&self, i: usize) -> u64 {
        self.reserve.base() + (i * self.page) as u64
    }

    /// Memory of its own for one page, the context bound to the thread.
    fn new_page(&self) -> Result<Arc<PhysicalAllocation>, GpuError> {
        let ctx = self.stream.context();
        ctx.bind_to_thread()?;
        let own =
            PhysicalAllocation::new(ctx.cu_device(), self.page).map_err(driver("cuMemCreate"))?;
        Ok(Arc::new(own))
    }

    /// The device's free bytes on this room's context, which the calling
    /// fit has bound to the thread, for its named refusal; the reading's
    /// own failure is that refusal's to name.
    fn free_bytes(&self) -> Result<usize, GpuError> {
        // The room's context is current on this thread: the fit's `new_page`
        // bound it before the refusal this reading names.
        Ok(crate::current_mem_info()?.0)
    }

    /// A fit's named refusal of page `i`, the driver's `source` in it: the
    /// store, the positions the fit asked, the bytes of new pages the fit
    /// still asked, the page among the room's, the pages the room had
    /// backed and the device's free bytes then.
    fn no_room(&self, positions: usize, i: usize, source: GpuError) -> GpuError {
        let want = self.wanted(positions);
        let asked = want
            .iter()
            .enumerate()
            .filter(|&(i, &w)| w && self.own(i).is_none())
            .count()
            .saturating_mul(self.page);
        let free = self.free_bytes().map_or_else(
            |e| format!("free bytes unreadable ({e})"),
            |free| format!("{free} B free"),
        );
        GpuError::NoRoom {
            what: self.what,
            detail: format!(
                "a fit of {positions} positions asks {asked} B of new pages (backing page {i} \
                 of {}, {} backed) and the device holds {free}",
                self.pages.len(),
                self.mapped
            ),
            source: Box::new(source),
        }
    }

    /// The memory page `i` maps, other than the spare.
    fn own(&self, i: usize) -> Option<&Arc<PhysicalAllocation>> {
        self.pages[i].as_ref().and_then(|p| p.own.as_ref())
    }

    /// Whether page `i` maps memory another room maps too.
    fn is_shared(&self, i: usize) -> bool {
        self.own(i).is_some_and(|o| Arc::strong_count(o) > 1)
    }

    /// Whether page `i` touches a run's rows and holds, in every run it
    /// touches, whole rows of the first `positions` positions alone: a page
    /// no write from `positions` on reaches.
    fn below(&self, i: usize, positions: usize) -> bool {
        let (start, end) = (i * self.page, (i + 1) * self.page);
        let mut touched = false;
        for r in &self.runs {
            let (at, last) = r.span(usize::MAX);
            if end <= at || start >= last {
                continue;
            }
            touched = true;
            if end.min(last) > r.written_from(positions) {
                return false;
            }
        }
        touched
    }

    /// Whether page `i` holds a row a write from position `pos` on reaches.
    fn written_from(&self, i: usize, pos: usize) -> bool {
        let (start, end) = (i * self.page, (i + 1) * self.page);
        self.runs.iter().any(|r| {
            let (_, last) = r.span(usize::MAX);
            start < last && end > r.written_from(pos)
        })
    }

    /// `bytes` bytes from `src` to `dst` on the room's stream, in order
    /// before any launch enqueued after.
    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        // SAFETY: both ranges are `bytes` bytes of device memory mapped
        // read-write for this device (pages of rooms of this device, or a
        // room's scratch mappings of new pages) that stay mapped until the
        // stream has run the copy, and they do not overlap.
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())
        }
        .map_err(driver("cuMemcpyDtoDAsync"))
    }

    /// Page `i`'s bytes zeroed on the room's stream, in order before any
    /// launch enqueued after.
    fn zero(&self, i: usize) -> Result<(), GpuError> {
        let va = self.va(i);
        // SAFETY: `va` is the start of a page of this room's reservation,
        // `page` bytes long, mapped read-write for this device; the room
        // keeps a mapping there until it drops, after the device finished.
        unsafe {
            cuda_core::simt::memory::memset_d8_async(va, 0, self.page, self.stream.cu_stream())
        }
        .map_err(driver("cuMemsetD8Async"))
    }

    /// The most positions whose rows lie on the pages the rows of the first
    /// `positions` positions take, in every run.
    fn covered(&self, positions: usize) -> usize {
        self.runs
            .iter()
            .map(|r| {
                let rows = positions.div_ceil(r.per).min(r.rows);
                if rows == r.rows {
                    return usize::MAX;
                }
                if rows == 0 {
                    return 0;
                }
                let end = (r.at + rows * r.row).div_ceil(self.page) * self.page;
                ((end - r.at) / r.row).saturating_mul(r.per)
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    /// Every page the rows of the first `positions` positions lie on backed
    /// by its own memory: those not yet are given it, zeroed. A fit within
    /// the backed pages returns at once (a decode step's, but on a page's
    /// first row). Never inside a capture.
    pub(crate) fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.reach = self.reach.max(positions);
        if positions <= self.fitted {
            return Ok(());
        }
        for r in 0..self.runs.len() {
            let (_, from) = self.runs[r].span(self.fitted);
            let (_, end) = self.runs[r].span(positions);
            if end == from {
                continue;
            }
            for i in from / self.page..end.div_ceil(self.page) {
                if self.own(i).is_some() {
                    continue;
                }
                let own = self
                    .new_page()
                    .map_err(|source| self.no_room(positions, i, source))?;
                self.back(i, Some(own))?;
                self.zero(i)?;
            }
        }
        self.fitted = self.covered(positions);
        Ok(())
    }

    /// The end of a sequence: every page none of the rows of the positions
    /// it reached lies on, and every page another room maps too, given back
    /// to the spare, after the device has finished every call that could
    /// read one, and the next sequence's reach counted from none.
    pub(crate) fn settle(&mut self) -> Result<(), GpuError> {
        let keep = self.wanted(self.reach);
        let give: Vec<usize> = (0..self.pages.len())
            .filter(|&i| self.own(i).is_some() && (!keep[i] || self.is_shared(i)))
            .collect();
        let reached = give.iter().any(|&i| keep[i]);
        if !give.is_empty() {
            crate::capsync::ctx_sync(self.stream.context(), "room settle")?;
            self.stream.context().bind_to_thread()?;
            for i in give {
                self.back(i, None)?;
            }
        }
        self.fitted = if reached {
            0
        } else {
            self.fitted.min(self.covered(self.reach))
        };
        self.reach = 0;
        Ok(())
    }

    /// Refused by name unless `from` is a room of this one's layout on its
    /// device and backs every page the rows of the first `positions`
    /// positions lie on: what [`Room::share`] takes.
    fn check_share(&self, from: &Room, positions: usize) -> Result<(), GpuError> {
        if self.page != from.page
            || self.runs != from.runs
            || self.pages.len() != from.pages.len()
            || self.stream.context().cu_device() != from.stream.context().cu_device()
        {
            return Err(GpuError::shape(
                WHAT,
                "a share from a room of another layout or device",
            ));
        }
        let want = from.wanted(positions);
        if let Some(i) = (0..want.len()).find(|&i| want[i] && from.own(i).is_none()) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a share of {positions} positions whose rows lie on page {i} of a room that \
                     backs no memory there"
                ),
            ));
        }
        Ok(())
    }

    /// The rows of the first `positions` positions taken from `from`, a room
    /// of the same layout on the same device: each page that holds whole
    /// rows of those positions alone ([`Room::below`]) maps `from`'s memory,
    /// and each other page one of their rows lies on gets memory of its own
    /// (unless it holds memory no other room maps) with `from`'s page copied
    /// in on the room's stream. The pages past the rows stay as they are,
    /// and the reach counts the positions. Synchronizes the context first:
    /// no call reads a page while its mapping moves. Refused by name,
    /// nothing moved, as [`Room::check_share`] refuses. Never inside a
    /// capture.
    pub(crate) fn share(&mut self, from: &Room, positions: usize) -> Result<SharedPages, GpuError> {
        self.check_share(from, positions)?;
        let want = self.wanted(positions);
        crate::capsync::ctx_sync(self.stream.context(), "room share")?;
        self.stream.context().bind_to_thread()?;
        let mut moved = SharedPages::default();
        self.lent.store(true, Ordering::Relaxed);
        from.lent.store(true, Ordering::Relaxed);
        for i in (0..want.len()).filter(|&i| want[i]) {
            let theirs = from
                .own(i)
                .ok_or(GpuError::state(WHAT, "the shared page's memory"))?;
            if self.below(i, positions) {
                if !self.own(i).is_some_and(|o| Arc::ptr_eq(o, theirs)) {
                    self.back(i, Some(Arc::clone(theirs)))?;
                }
                moved.mapped += self.page;
            } else {
                if self.own(i).is_none() || self.is_shared(i) {
                    let own = self.new_page()?;
                    self.back(i, Some(own))?;
                }
                self.copy(self.va(i), from.va(i), self.page)?;
                moved.copied += self.page;
            }
        }
        self.fitted = self.fitted.max(self.covered(positions));
        self.reach = self.reach.max(positions);
        Ok(moved)
    }

    /// Every page a write from position `pos` on reaches that maps memory
    /// another room maps too given memory of its own holding the same bytes:
    /// what a writer calls before it writes rows from `pos`, so no write
    /// reaches a page another sequence reads. The new pages are made and
    /// filled first through mappings of their own in a scratch range, so a
    /// failure leaves every page as it was and the copy takes no memory
    /// beyond the pages themselves. The context is synchronized before the
    /// mappings move and the copies are waited for; nothing happens when no
    /// such page stands. Never inside a capture.
    pub(crate) fn own_from(&mut self, pos: usize) -> Result<(), GpuError> {
        if !self.lends() {
            return Ok(());
        }
        let moving: Vec<usize> = (0..self.pages.len())
            .filter(|&i| self.is_shared(i) && self.written_from(i, pos))
            .collect();
        if moving.is_empty() {
            return Ok(());
        }
        crate::capsync::ctx_sync(self.stream.context(), "room own")?;
        self.stream.context().bind_to_thread()?;
        let scratch = VirtualReservation::new(moving.len() * self.page, self.page)
            .map_err(driver("cuMemAddressReserve"))?;
        // Each entry's mapping drops before its memory, and all before the
        // scratch range.
        let mut fresh = Vec::with_capacity(moving.len());
        for (k, &i) in moving.iter().enumerate() {
            let own = self.new_page()?;
            let va = scratch.base() + (k * self.page) as u64;
            let map = Mapping::new(va, self.page, &own, 0).map_err(driver("cuMemMap"))?;
            vmm::set_access(va, self.page, &[self.stream.context().cu_device()])
                .map_err(driver("cuMemSetAccess"))?;
            fresh.push((i, map, own, va));
        }
        let sent =
            (fresh.iter()).try_for_each(|&(i, _, _, va)| self.copy(va, self.va(i), self.page));
        let waited = self.stream.synchronize();
        sent?;
        waited?;
        for (i, map, own, _) in fresh {
            drop(map);
            self.back(i, Some(own))?;
        }
        Ok(())
    }

    /// Device bytes the pages with memory of their own hold, those that
    /// map another room's memory too among them.
    pub(crate) fn mapped(&self) -> usize {
        self.mapped * self.page
    }

    /// Device bytes of the pages that map memory another room maps too.
    pub(crate) fn shared(&self) -> usize {
        if !self.lends() {
            return 0;
        }
        (0..self.pages.len()).filter(|&i| self.is_shared(i)).count() * self.page
    }

    /// Whether a page maps memory another room maps too, scanned only while
    /// the room's `lent` flag stands; a scan that finds none clears it.
    fn lends(&self) -> bool {
        if !self.lent.load(Ordering::Relaxed) {
            return false;
        }
        let any = (0..self.pages.len()).any(|i| self.is_shared(i));
        if !any {
            self.lent.store(false, Ordering::Relaxed);
        }
        any
    }

    /// Device bytes every page of the range holds once backed: what a fill
    /// takes.
    pub(crate) fn whole(&self) -> usize {
        self.pages.len() * self.page
    }
}

impl Drop for Room {
    fn drop(&mut self) {
        // Pages unmap after this body; keep a failed drain on the context.
        self.stream
            .context()
            .record_err(crate::capsync::ctx_sync_in_drop(
                self.stream.context(),
                "room drop",
            ));
    }
}

/// What a room's buffers are held as: every buffer taken apart without a
/// free when the holder drops, since its memory is the room's.
pub(crate) trait Views {
    fn dissolve(self);
}

impl<T> Views for DeviceBuffer<T> {
    fn dissolve(self) {
        drop(self.into_raw_parts());
    }
}

impl<T, const N: usize> Views for [DeviceBuffer<T>; N] {
    fn dissolve(self) {
        self.into_iter().for_each(Views::dissolve);
    }
}

/// Buffers over one or more [`Room`]s and the rooms: the buffers read and
/// write as any other (`Deref`), the rooms back their pages
/// ([`Roomed::fit`]).
pub(crate) struct Roomed<V: Views> {
    views: ManuallyDrop<V>,
    rooms: Vec<Room>,
}

impl<V: Views> Roomed<V> {
    /// `N` buffers of `len` elements in one reserve on `stream`'s device,
    /// each at a page boundary, built into `V` by `build`; `runs` are the
    /// rows of one buffer, which every buffer has at its own offset. Every
    /// page maps the spare: the rows a fit reaches get memory of their own,
    /// zeroed ([`Room::fit`]). `what` names the store the room backs, for
    /// a fit's named refusal. Refused by name for a run past a buffer.
    pub(crate) fn new<T, const N: usize>(
        stream: Arc<CudaStream>,
        len: usize,
        runs: &[Run],
        what: &'static str,
        build: impl FnOnce([DeviceBuffer<T>; N]) -> V,
    ) -> Result<Roomed<V>, GpuError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::shape(WHAT, format!("{len} elements pass usize")))?;
        if let Some(r) = runs.iter().find(|r| r.at + r.rows * r.row > bytes) {
            return Err(GpuError::shape(
                WHAT,
                format!("{r:?} past a buffer of {bytes} bytes"),
            ));
        }
        let ctx = Arc::clone(stream.context());
        ctx.bind_to_thread()?;
        let page = crate::allocation_granularity(ctx.ordinal())?;
        let stride = vmm::align_size(bytes.max(1), page);
        let reserve =
            VirtualReservation::new(stride * N, page).map_err(driver("cuMemAddressReserve"))?;
        let (spare, fresh) = spare(ctx.cu_device(), page)?;
        let base = reserve.base();
        let mut room = Room {
            what,
            pages: (0..N * stride / page).map(|_| None).collect(),
            runs: (0..N)
                .flat_map(|i| {
                    runs.iter().map(move |r| Run {
                        at: r.at + i * stride,
                        ..*r
                    })
                })
                .collect(),
            fitted: 0,
            reach: 0,
            mapped: 0,
            lent: AtomicBool::new(false),
            page,
            spare,
            stream,
            reserve,
        };
        for i in 0..room.pages.len() {
            room.back(i, None)?;
        }
        if fresh {
            room.zero(0)?;
        }
        // The views last: an error before them returns with no buffer over
        // the range, so none frees an address in it.
        let views = std::array::from_fn(|i| {
            // SAFETY: the buffer lies in this room's reservation, which the
            // `Roomed` below owns and outlives it: `Roomed::drop` takes every
            // buffer apart without a free (`Views::dissolve`) before the room
            // drops, so no `cuMemFree` ever reaches these addresses.
            unsafe {
                DeviceBuffer::from_raw_parts(base + (i * stride) as u64, len, Arc::clone(&ctx))
            }
        });
        Ok(Roomed {
            views: ManuallyDrop::new(build(views)),
            rooms: vec![room],
        })
    }

    /// The same buffers and rooms held as `W`, built by `build`.
    pub(crate) fn map<W: Views>(self, build: impl FnOnce(V) -> W) -> Roomed<W> {
        let (views, rooms) = self.into_parts();
        Roomed {
            views: ManuallyDrop::new(build(views)),
            rooms,
        }
    }

    /// The buffers of `a` and `b` built into one `V` by `build`, over both
    /// rooms: buffers of two element types or lengths in one holder.
    pub(crate) fn join<A: Views, B: Views>(
        a: Roomed<A>,
        b: Roomed<B>,
        build: impl FnOnce(A, B) -> V,
    ) -> Roomed<V> {
        let (a, mut rooms) = a.into_parts();
        let (b, more) = b.into_parts();
        rooms.extend(more);
        Roomed {
            views: ManuallyDrop::new(build(a, b)),
            rooms,
        }
    }

    /// The views and the rooms, the holder gone without dissolving them.
    fn into_parts(mut self) -> (V, Vec<Room>) {
        let rooms = std::mem::take(&mut self.rooms);
        // SAFETY: the views are taken once, here; the holder is forgotten
        // below, so its `Drop` never takes them again.
        let views = unsafe { ManuallyDrop::take(&mut self.views) };
        std::mem::forget(self);
        (views, rooms)
    }

    /// The rooms the buffers lie in.
    pub(crate) fn rooms_mut(&mut self) -> &mut [Room] {
        &mut self.rooms
    }

    /// Every page the rows of the first `positions` positions lie on backed
    /// by its own memory, zeroed ([`Room::fit`]). Never inside a capture.
    pub(crate) fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.rooms.iter_mut().try_for_each(|r| r.fit(positions))
    }

    /// Every row backed: what a read or write of the whole buffers takes.
    pub(crate) fn fill(&mut self) -> Result<(), GpuError> {
        self.fit(usize::MAX)
    }

    /// A sequence's end: the memory of the pages past the rows it reached,
    /// and of the pages another room maps too, given back ([`Room::settle`]).
    pub(crate) fn settle(&mut self) -> Result<(), GpuError> {
        self.rooms.iter_mut().try_for_each(Room::settle)
    }

    /// The rows of the first `positions` positions taken from `from`'s
    /// buffers, room by room ([`Room::share`]). Refused by name, nothing
    /// moved, unless every room may take its share ([`Room::check_share`]).
    /// Never inside a capture.
    pub(crate) fn share(
        &mut self,
        from: &Roomed<V>,
        positions: usize,
    ) -> Result<SharedPages, GpuError> {
        if self.rooms.len() != from.rooms.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a share from buffers over {} rooms into buffers over {}",
                    from.rooms.len(),
                    self.rooms.len()
                ),
            ));
        }
        let pairs = || self.rooms.iter().zip(&from.rooms);
        pairs().try_for_each(|(r, f)| r.check_share(f, positions))?;
        (self.rooms.iter_mut().zip(&from.rooms)).try_fold(SharedPages::default(), |m, (r, f)| {
            Ok(m + r.share(f, positions)?)
        })
    }

    /// Every page a write from position `pos` on reaches given memory no
    /// other room maps ([`Room::own_from`]). Never inside a capture.
    pub(crate) fn own_from(&mut self, pos: usize) -> Result<(), GpuError> {
        self.rooms.iter_mut().try_for_each(|r| r.own_from(pos))
    }

    /// Device bytes the pages with memory of their own hold.
    pub(crate) fn mapped(&self) -> usize {
        self.rooms.iter().map(Room::mapped).sum()
    }

    /// Device bytes of the pages that map memory another room maps too
    /// ([`Room::shared`]).
    pub(crate) fn shared(&self) -> usize {
        self.rooms.iter().map(Room::shared).sum()
    }

    /// Device bytes every page holds once backed ([`Room::whole`]).
    pub(crate) fn whole(&self) -> usize {
        self.rooms.iter().map(Room::whole).sum()
    }
}

impl<V: Views> Deref for Roomed<V> {
    type Target = V;
    fn deref(&self) -> &V {
        &self.views
    }
}

impl<V: Views> DerefMut for Roomed<V> {
    fn deref_mut(&mut self) -> &mut V {
        &mut self.views
    }
}

impl<V: Views> Drop for Roomed<V> {
    fn drop(&mut self) {
        // SAFETY: the views are taken once, here, and never touched again;
        // the rooms drop after this body, unmapping what they addressed.
        unsafe { ManuallyDrop::take(&mut self.views) }.dissolve();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Pitched, Window, WindowMut, dtoh_2d, htod_2d};

    /// A room backs only the pages a fit reaches, zeroed, at the addresses
    /// reserved for the whole buffers: two buffers of two heads' rows, four
    /// rows a page and eighteen rows a head, put the second head half a
    /// page past a boundary: fits of one, four and five reach different
    /// page boundaries in the two heads;
    /// values written to a backed row read back and a page just backed
    /// reads zero; a settle keeps every page the reach needs and gives the
    /// rest back at the next one; a 2-D copy of each head's first row,
    /// whose extent crosses the unbacked pages between, writes and reads;
    /// a fill backs every page; a run past its buffer is refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_room_backs_the_rows_a_fit_reaches() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, ctx_rows) = (page / 4, 18);
        let half = row / size_of::<u16>();
        let runs: Vec<Run> = (0..2)
            .map(|h| Run {
                at: h * ctx_rows * row,
                row,
                per: 1,
                rows: ctx_rows,
            })
            .collect();
        let len = 2 * ctx_rows * half;
        let mut r: Roomed<[DeviceBuffer<u16>; 2]> =
            Roomed::new(Arc::clone(&stream), len, &runs, "the tests' planes", |b| b)
                .expect("a room");
        let at = (r[0].cu_deviceptr(), r[1].cu_deviceptr());
        assert_eq!(r.mapped(), 0, "no page before a fit");
        assert_eq!(r.whole(), 18 * page, "every page of both buffers");
        for (positions, pages) in [(1, 4), (4, 6), (5, 8)] {
            r.fit(positions).expect("a fit");
            assert_eq!(r.mapped(), pages * page, "a fit of {positions} positions");
        }
        let head1_row4 = (ctx_rows + 4) * row;
        WindowMut::<u16>::of_mut(&mut r[0], head1_row4, half)
            .expect("a backed row")
            .copy_from_host(&stream, &vec![7u16; half])
            .expect("a write");
        let read = |b: &DeviceBuffer<u16>, byte: usize| {
            Window::<u16>::of(b, byte, half)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")
        };
        assert_eq!(read(&r[0], head1_row4), vec![7u16; half]);
        assert_eq!(
            read(&r[1], 4 * row),
            vec![0u16; half],
            "a page just mapped reads zero"
        );
        r.settle().expect("a settle");
        assert_eq!(
            r.mapped(),
            8 * page,
            "the reach of five keeps both pages of each head"
        );
        r.fit(1).expect("a fit");
        r.settle().expect("a settle");
        assert_eq!(r.mapped(), 4 * page, "a reach of one keeps the first pages");
        let heads = Pitched {
            width: row,
            rows: 2,
            dev_pitch: ctx_rows * row,
            host_pitch: row,
        };
        let put = vec![3u16; 2 * half];
        let mut got = vec![0u16; 2 * half];
        // SAFETY: `put` holds the copy's `2 * row` bytes and is neither
        // written nor freed before the stream's synchronize below.
        unsafe { htod_2d(&stream, &mut r[1], 0, put.as_ptr().cast(), 2 * row, heads) }
            .expect("a 2-D write of each head's first row over the unbacked pages between");
        // SAFETY: `got` holds the copy's `2 * row` bytes and is neither read
        // nor freed before the stream's synchronize below.
        unsafe { dtoh_2d(&stream, &r[1], 0, got.as_mut_ptr().cast(), 2 * row, heads) }
            .expect("a 2-D read of each head's first row over the unbacked pages between");
        stream.synchronize().expect("the copies");
        assert_eq!(got, put);
        r.fill().expect("a fill");
        assert_eq!(r.mapped(), r.whole(), "a fill backs every page");
        assert_eq!(
            read(&r[0], head1_row4),
            vec![0u16; half],
            "a page given back and backed again reads zero"
        );
        assert_eq!(
            (r[0].cu_deviceptr(), r[1].cu_deviceptr()),
            at,
            "the addresses stay"
        );
        let past = [Run {
            rows: ctx_rows + 1,
            ..runs[1]
        }];
        let Err(e) = Roomed::<[DeviceBuffer<u16>; 2]>::new(
            Arc::clone(&stream),
            len,
            &past,
            "the tests' planes",
            |b| b,
        ) else {
            panic!("a run past its buffer is refused");
        };
        assert!(e.to_string().contains("past a buffer"), "{e}");
        let pooled = [Run {
            at: 0,
            row: page,
            per: 4,
            rows: 4,
        }];
        let mut p: Roomed<[DeviceBuffer<u16>; 1]> = Roomed::new(
            Arc::clone(&stream),
            4 * page / 2,
            &pooled,
            "the tests' pooled keys",
            |b| b,
        )
        .expect("a room");
        for (positions, pages) in [(1, 1), (4, 1), (5, 2)] {
            p.fit(positions).expect("a fit");
            assert_eq!(
                p.mapped(),
                pages * page,
                "a pooled fit of {positions} positions"
            );
        }
        stream.synchronize().expect("the room's work finished");
        let drain_error = DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_UNKNOWN);
        ctx.record_err::<()>(Err(drain_error));
        drop(p);
        assert_eq!(
            ctx.check_err(),
            Err(drain_error),
            "a room's failed drain stays on its retained context"
        );
    }

    /// A fit past the device's free bytes is refused by name: the store, the
    /// positions, the bytes of new pages the fit asks, the page it was
    /// backing among the room's, the pages the room had backed and the
    /// device's free bytes, with the driver's own refusal. With the device
    /// holding all but a few pages free, the first pages of a fit back and a
    /// later one cannot.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_fit_past_the_free_bytes_is_refused_by_name() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        ctx.bind_to_thread().expect("the context binds");
        let (free, _) = crate::current_mem_info().expect("the device's free bytes");
        let held = DeviceBuffer::<u8>::zeroed(&stream, free.saturating_sub(6 * page) & !(page - 1))
            .expect("the device's free bytes less a few pages");
        let runs = [Run {
            at: 0,
            row: page,
            per: 1,
            rows: 16,
        }];
        let mut r: Roomed<[DeviceBuffer<u16>; 1]> = Roomed::new(
            Arc::clone(&stream),
            16 * page / 2,
            &runs,
            "the refusal's store",
            |b| b,
        )
        .expect("a room of 16 pages");
        let Err(e) = r.fit(16) else {
            panic!("a fit past the free bytes is refused");
        };
        let text = e.to_string();
        assert!(text.contains("the refusal's store"), "{text}");
        assert!(text.contains("a fit of 16 positions"), "{text}");
        assert!(text.contains("asks "), "{text}");
        assert!(
            text.contains("backing page ") && text.contains(" of 16, "),
            "{text}"
        );
        assert!(text.contains(" B free"), "{text}");
        assert!(text.contains("cuMemCreate"), "{text}");
        drop(r);
        drop(held);
        stream.synchronize().expect("the room's work finished");
    }

    /// Two rooms of one layout share a prefix's pages: the layout of
    /// [`hw_a_room_backs_the_rows_a_fit_reaches`] (four rows a page, the
    /// second head half a page past a boundary), the source's first thirteen
    /// rows of every head written, a share of nine positions maps the three
    /// pages of each buffer that hold whole rows below nine alone and copies
    /// the three others those rows lie on; the rows read back as the
    /// source's; a write from nine on moves nothing, a write from five on
    /// gives the two pages of each buffer it reaches memory of their own with
    /// the same bytes, after which a write there leaves the source's row as
    /// it was; a settle gives the shared pages back and the source keeps its
    /// rows; a room of another layout, and a source that backs no page
    /// under the rows, are refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_rooms_share_the_rows_below_a_point() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, ctx_rows) = (page / 4, 18);
        let half = row / size_of::<u16>();
        let layout = |rows: usize| -> Vec<Run> {
            (0..2)
                .map(|h| Run {
                    at: h * rows * row,
                    row,
                    per: 1,
                    rows,
                })
                .collect()
        };
        let room = |rows: usize| -> Roomed<[DeviceBuffer<u16>; 2]> {
            Roomed::new(
                Arc::clone(&stream),
                2 * rows * half,
                &layout(rows),
                "the tests' planes",
                |b| b,
            )
            .expect("a room")
        };
        let byte = |h: usize, j: usize| (h * ctx_rows + j) * row;
        let value = |b: usize, h: usize, j: usize| (b * 100 + h * 40 + j + 1) as u16;
        let write = |r: &mut Roomed<[DeviceBuffer<u16>; 2]>, b: usize, h: usize, j: usize, v| {
            WindowMut::<u16>::of_mut(&mut r[b], byte(h, j), half)
                .expect("a backed row")
                .copy_from_host(&stream, &vec![v; half])
                .expect("a write");
        };
        let read = |r: &Roomed<[DeviceBuffer<u16>; 2]>, b: usize, h: usize, j: usize| {
            Window::<u16>::of(&r[b], byte(h, j), half)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")[0]
        };
        let rows_of = |r: &Roomed<[DeviceBuffer<u16>; 2]>, n: usize| -> Vec<u16> {
            (0..2)
                .flat_map(|b| (0..2).flat_map(move |h| (0..n).map(move |j| (b, h, j))))
                .map(|(b, h, j)| read(r, b, h, j))
                .collect()
        };
        let mut a = room(ctx_rows);
        a.fit(13).expect("a fit");
        for b in 0..2 {
            for h in 0..2 {
                for j in 0..13 {
                    write(&mut a, b, h, j, value(b, h, j));
                }
            }
        }
        let mut b = room(ctx_rows);
        b.fit(2).expect("a fit");
        let moved = b.share(&a, 9).expect("a share");
        assert_eq!(
            moved,
            SharedPages {
                mapped: 6 * page,
                copied: 6 * page
            },
            "pages 0, 1, 5 of each buffer mapped, 2, 4, 6 copied"
        );
        assert_eq!((a.shared(), b.shared()), (6 * page, 6 * page));
        assert_eq!(rows_of(&b, 9), rows_of(&a, 9), "the shared rows");
        a.own_from(9).expect("a write from 9");
        assert_eq!(
            a.shared(),
            6 * page,
            "a write from 9 reaches no shared page"
        );
        b.own_from(5).expect("a write from 5");
        assert_eq!(
            (a.shared(), b.shared()),
            (2 * page, 2 * page),
            "pages 1 and 5 of each buffer given their own memory"
        );
        assert_eq!(rows_of(&b, 9), rows_of(&a, 9), "the bytes moved with them");
        write(&mut b, 0, 0, 4, 999);
        assert_eq!(read(&b, 0, 0, 4), 999);
        assert_eq!(read(&a, 0, 0, 4), value(0, 0, 4), "the source's row stands");
        b.settle().expect("a settle");
        assert_eq!(
            (a.shared(), b.shared()),
            (0, 0),
            "the shared pages given back"
        );
        let want: Vec<u16> = (0..2)
            .flat_map(|b| (0..2).flat_map(move |h| (0..13).map(move |j| value(b, h, j))))
            .collect();
        assert_eq!(rows_of(&a, 13), want, "the source keeps its rows");
        let other = room(ctx_rows + 4);
        let e = b.share(&other, 4).expect_err("another layout").to_string();
        assert!(e.contains("another layout"), "{e}");
        let bare = room(ctx_rows);
        let e = b
            .share(&bare, 4)
            .expect_err("an unbacked source")
            .to_string();
        assert!(e.contains("backs no memory"), "{e}");
        stream.synchronize().expect("the rooms' work finished");
    }
}
