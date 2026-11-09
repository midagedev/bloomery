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
//! A routed expert stack a residency machine runs over is a [`StackRoom`]: its
//! range reserved whole and every page backed at load, so it reads and
//! launches as a plain buffer until the machine gives back the pages wholly
//! past a slot ([`StackRoom::release_from`]) and backs them again
//! ([`StackRoom::restore`]) — a stack's pages are always card memory
//! ([`Backing::Card`]).
//!
//! A room's own memory can sit on the host instead of the card
//! ([`Backing`]): a host page is mapped and granted to the device exactly as
//! a card page and zeroed on the room's stream, so every kernel and capture
//! over the store stays as it is; [`Room::mapped`] keeps naming the card
//! bytes and [`Room::host_mapped`] names the host bytes. A [`Backing::Spill`]
//! room backs a page on the host when the card's free bytes would fall below
//! its margin by the page, or when the card cannot make the page's memory for
//! lack of free bytes.
//!
//! Every writer calls `fit` and `own_from` first, outside capture, for all
//! positions it can write. An unbacked page aliases the spare: a missed fit
//! can corrupt another room without a device fault, and a missed `own_from`
//! another sequence's rows.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use cuda_core::sys::CUdevice;
use cuda_core::vmm::{self, PhysicalAllocation, VirtualReservation};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::GpuError;
use crate::tensor::DeviceTensor;

pub(crate) mod elastic;
pub(crate) mod host;
pub(crate) mod kv;

use self::host::{HostPage, RawMap};

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

/// The memory a room backs its pages with, chosen at construction and held
/// by every page the room gives memory of its own to: [`Room::new_page`]
/// makes each page's memory where this says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backing {
    /// The device's own memory: today's only choice.
    Card,
    /// Host memory on the NUMA node the device names, mapped into the range
    /// and reached over the device's link ([`host::HostPage`]).
    Host,
    /// The card's memory first, keeping `margin` bytes of the card free: a
    /// page whose card memory would leave fewer free bytes than that, or
    /// that the driver cannot make for lack of free bytes, is backed on the
    /// host instead. Any other driver refusal stays the named refusal it is.
    Spill { margin: u64 },
}

/// The memory a page backs: the device's own, or host memory the device
/// reaches over its link. One `Arc` a page; a share hands another room the
/// same one (the `Clone` is the `Arc`'s hold).
#[derive(Clone)]
enum Owned {
    Card(Arc<PhysicalAllocation>),
    Host(Arc<HostPage>),
}

impl Owned {
    /// The raw allocation handle, for a mapping.
    fn handle(&self) -> cuda_core::sys::CUmemGenericAllocationHandle {
        match self {
            Owned::Card(a) => a.handle(),
            Owned::Host(h) => h.handle(),
        }
    }

    /// Whether the memory sits on the host.
    fn is_host(&self) -> bool {
        matches!(self, Owned::Host(_))
    }

    /// Rooms that hold this memory, the page's own room among them.
    fn holders(&self) -> usize {
        match self {
            Owned::Card(a) => Arc::strong_count(a),
            Owned::Host(h) => Arc::strong_count(h),
        }
    }

    /// Whether two pages map the same memory.
    fn same(a: &Owned, b: &Owned) -> bool {
        match (a, b) {
            (Owned::Card(x), Owned::Card(y)) => Arc::ptr_eq(x, y),
            (Owned::Host(x), Owned::Host(y)) => Arc::ptr_eq(x, y),
            _ => false,
        }
    }
}

/// A page of a room's range: its mapping, and the memory it maps other
/// than the spare (its own, or another room's too after a share). The
/// mapping drops first (unmapped), then this room's hold on the memory.
struct Page {
    _map: RawMap,
    own: Option<Owned>,
}

/// The page every page of `device`'s rooms of `page` bytes with no memory
/// of its own maps, and whether it was made just now (its bytes not set
/// yet). One a device and page size while any room holds it — a room that
/// can back pages on the host may take a larger page than a card-only room
/// ([`host::page_for`]). Its bytes are no row's: nothing reads or writes
/// them but an access past the rows in use.
fn spare(device: CUdevice, page: usize) -> Result<(Arc<PhysicalAllocation>, bool), GpuError> {
    static SPARES: Mutex<Vec<(CUdevice, usize, Weak<PhysicalAllocation>)>> = Mutex::new(Vec::new());
    let mut spares = SPARES.lock().unwrap_or_else(PoisonError::into_inner);
    spares.retain(|(_, _, s)| s.strong_count() > 0);
    if let Some(s) = spares
        .iter()
        .find(|(d, p, _)| *d == device && *p == page)
        .and_then(|(_, _, s)| s.upgrade())
    {
        return Ok((s, false));
    }
    let s = Arc::new(PhysicalAllocation::new(device, page).map_err(driver("cuMemCreate"))?);
    spares.push((device, page, Arc::downgrade(&s)));
    Ok((s, true))
}

fn driver(op: &'static str) -> impl Fn(DriverError) -> GpuError {
    move |source| GpuError::Driver {
        op: Some(op),
        source,
    }
}

/// Whether a driver refusal is the device's own memory running out: the one
/// a [`Backing::Spill`] room answers on the host.
fn is_out_of_memory(source: &DriverError) -> bool {
    source.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY
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
    /// Where a fit's pages get their memory ([`Backing`]).
    backing: Backing,
    /// The positions whose rows lie on pages with memory of their own: a
    /// fit up to it backs nothing new.
    fitted: usize,
    /// The most positions a fit asked since the last settle.
    reach: usize,
    /// The pages with card memory of their own.
    mapped: usize,
    /// The pages with host memory of their own.
    hosted: usize,
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
    fn back(&mut self, i: usize, own: Option<Owned>) -> Result<(), GpuError> {
        if let Some(gone) = self.pages[i].take() {
            match gone.own {
                Some(Owned::Card(_)) => self.mapped -= 1,
                Some(Owned::Host(_)) => self.hosted -= 1,
                None => {}
            }
        }
        let va = self.va(i);
        let handle = own
            .as_ref()
            .map_or_else(|| self.spare.handle(), Owned::handle);
        let map = RawMap::new(va, self.page, handle, 0)?;
        vmm::set_access(va, self.page, &[self.stream.context().cu_device()])
            .map_err(driver("cuMemSetAccess"))?;
        match own {
            Some(Owned::Card(_)) => self.mapped += 1,
            Some(Owned::Host(_)) => self.hosted += 1,
            None => {}
        }
        self.pages[i] = Some(Page { _map: map, own });
        Ok(())
    }

    /// Page `i`'s address.
    fn va(&self, i: usize) -> u64 {
        self.reserve.base() + (i * self.page) as u64
    }

    /// Memory of its own for one page, where the room's backing says
    /// ([`Backing`]), the context bound to the thread. A `Spill` room backs
    /// the page on the host when the card's free bytes are under its margin
    /// plus the page, or when the driver cannot make the card memory for
    /// lack of free bytes; any other driver refusal stays the refusal it is.
    fn new_page(&self) -> Result<Owned, GpuError> {
        let ctx = self.stream.context();
        ctx.bind_to_thread()?;
        let device = ctx.cu_device();
        match self.backing {
            Backing::Card => Ok(Owned::Card(Arc::new(
                PhysicalAllocation::new(device, self.page).map_err(driver("cuMemCreate"))?,
            ))),
            Backing::Host => Ok(Owned::Host(Arc::new(HostPage::new(device, self.page)?))),
            Backing::Spill { margin } => {
                let host = || -> Result<Owned, GpuError> {
                    Ok(Owned::Host(Arc::new(HostPage::new(device, self.page)?)))
                };
                if !self.keeps_margin(margin)? {
                    return host();
                }
                match PhysicalAllocation::new(device, self.page) {
                    Ok(card) => Ok(Owned::Card(Arc::new(card))),
                    Err(source) if is_out_of_memory(&source) => host(),
                    Err(source) => Err(driver("cuMemCreate")(source)),
                }
            }
        }
    }

    /// Whether a card page leaves the device at least `margin` bytes free,
    /// the context bound to the thread.
    fn keeps_margin(&self, margin: u64) -> Result<bool, GpuError> {
        Ok(self.free_bytes()? as u64 >= margin.saturating_add(self.page as u64))
    }

    /// The device's free bytes on this room's context, which the calling
    /// fit has bound to the thread, for a spill's margin and a fit's named
    /// refusal; the reading's own failure is that refusal's to name.
    fn free_bytes(&self) -> Result<usize, GpuError> {
        // The room's context is current on this thread: the fit's `new_page`
        // bound it before the refusal this reading names.
        Ok(crate::current_mem_info()?.0)
    }

    /// The bytes of new pages a fit of `positions` positions still asks: the
    /// pages its rows lie on that map no memory of their own.
    pub(crate) fn asks(&self, positions: usize) -> usize {
        self.wanted(positions)
            .iter()
            .enumerate()
            .filter(|&(i, &w)| w && self.own(i).is_none())
            .count()
            .saturating_mul(self.page)
    }

    /// A fit's named refusal of page `i`, the driver's `source` in it: the
    /// store, the positions the fit asked, the bytes of new pages the fit
    /// still asked, the page among the room's, the pages the room had backed
    /// on the card and on the host (the host bytes named as bytes, what a
    /// caller that spills reads) and the device's free bytes then. The asked
    /// bytes, the free bytes (when readable) and the page are the error's
    /// numbers.
    fn no_room(&self, positions: usize, i: usize, source: GpuError) -> GpuError {
        let asked = self.asks(positions);
        let free = self.free_bytes();
        let said = match &free {
            Ok(free) => format!("{free} B free"),
            Err(e) => format!("free bytes unreadable ({e})"),
        };
        GpuError::NoRoom {
            what: self.what,
            detail: format!(
                "a fit of {positions} positions asks {asked} B of new pages (backing page {i} \
                 of {}, {} card-backed, {} host-backed, {} B of host memory) and the device \
                 holds {said}",
                self.pages.len(),
                self.mapped,
                self.hosted,
                self.hosted * self.page
            ),
            source: Box::new(source),
            asked,
            free: free.ok(),
            page: self.page,
            spill: matches!(self.backing, Backing::Spill { .. }),
        }
    }

    /// The memory page `i` maps, other than the spare.
    fn own(&self, i: usize) -> Option<&Owned> {
        self.pages[i].as_ref().and_then(|p| p.own.as_ref())
    }

    /// Whether page `i` maps memory another room maps too.
    fn is_shared(&self, i: usize) -> bool {
        self.own(i).is_some_and(|o| o.holders() > 1)
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
    /// before any launch enqueued after. Either side may be a host-backed
    /// page: both are device-accessible addresses all the same.
    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        // SAFETY: both ranges are `bytes` bytes of device-accessible memory
        // (card or host pages) mapped read-write for this device (pages of
        // rooms of this device, or a room's scratch mappings of new pages)
        // that stay mapped until the stream has run the copy, and they do
        // not overlap.
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())
        }
        .map_err(driver("cuMemcpyDtoDAsync"))
    }

    /// Page `i`'s bytes zeroed on the room's stream, in order before any
    /// launch enqueued after.
    fn zero(&self, i: usize) -> Result<(), GpuError> {
        self.zero_on(i, &self.stream)
    }

    /// Page `i`'s bytes zeroed on `stream`, a stream of the room's context,
    /// in order before any work enqueued on it after.
    fn zero_on(&self, i: usize, stream: &CudaStream) -> Result<(), GpuError> {
        let va = self.va(i);
        // SAFETY: `va` is the start of a page of this room's reservation,
        // `page` bytes long, mapped read-write for this device; the room
        // keeps a mapping there until it drops, after the device finished.
        unsafe { cuda_core::simt::memory::memset_d8_async(va, 0, self.page, stream.cu_stream()) }
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

    /// The device bytes [`Room::restore`]`(from, to)` would back: the pages
    /// that hold a byte of `[from, to)` and map the spare.
    pub(crate) fn unbacked_bytes(&self, from: usize, to: usize) -> u64 {
        let end = to.div_ceil(self.page).min(self.pages.len());
        let pages = (from / self.page..end)
            .filter(|&i| self.own(i).is_none())
            .count();
        (pages * self.page) as u64
    }

    /// The device bytes of the pages wholly at or past byte `byte` of the
    /// range that hold card memory of their own: what [`Room::release_from`]
    /// would free. A host-backed page frees host bytes, not device bytes,
    /// and counts for none here.
    pub(crate) fn tail_bytes(&self, byte: usize) -> u64 {
        let pages = (byte.div_ceil(self.page)..self.pages.len())
            .filter(|&i| self.own(i).is_some_and(|o| !o.is_host()))
            .count();
        (pages * self.page) as u64
    }

    /// Every page wholly at or past byte `byte` of the range given back to
    /// the spare, its memory freed: the device bytes freed (the card-backed
    /// pages among them counted; a host-backed page frees host bytes, not
    /// device bytes), 0 for pages already on the spare. The caller has the
    /// device finished with every call that could read those pages (a copy
    /// out of them included), and nothing reads or writes them after. Never
    /// inside a capture. An error leaves the pages given back so far given
    /// back.
    pub(crate) fn release_from(&mut self, byte: usize) -> Result<u64, GpuError> {
        let give: Vec<usize> = (byte.div_ceil(self.page)..self.pages.len())
            .filter(|&i| self.own(i).is_some())
            .collect();
        if give.is_empty() {
            return Ok(0);
        }
        self.stream.context().bind_to_thread()?;
        let mut card = 0;
        for &i in &give {
            let host = self.own(i).is_some_and(|o| o.is_host());
            if !host {
                card += 1;
            }
            self.back(i, None)?;
        }
        Ok((card * self.page) as u64)
    }

    /// Memory of its own, zeroed on `stream` (a stream of the room's
    /// context), for every page that holds a byte of `[from, to)` and maps
    /// the spare: the pages [`Room::release_from`] gave back, the device
    /// bytes backed (the card-backed pages among them counted). The zeroing
    /// is in stream order: work enqueued on `stream` after this sees zeros.
    /// An error leaves the pages backed so far backed.
    pub(crate) fn restore(
        &mut self,
        from: usize,
        to: usize,
        stream: &CudaStream,
    ) -> Result<u64, GpuError> {
        let end = to.div_ceil(self.page).min(self.pages.len());
        let want: Vec<usize> = (from / self.page..end)
            .filter(|&i| self.own(i).is_none())
            .collect();
        let mut card = 0;
        for &i in &want {
            let own = self.new_page()?;
            if !own.is_host() {
                card += 1;
            }
            self.back(i, Some(own))?;
            self.zero_on(i, stream)?;
        }
        Ok((card * self.page) as u64)
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
                if !self.own(i).is_some_and(|o| Owned::same(o, theirs)) {
                    self.back(i, Some(theirs.clone()))?;
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
            let map = RawMap::new(va, self.page, own.handle(), 0)?;
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

    /// Card bytes the pages with card memory of their own hold, those that
    /// map another room's card memory too among them ([`Room::host_mapped`]
    /// names the host bytes; the plan and the residency machine read this as
    /// device bytes).
    pub(crate) fn mapped(&self) -> usize {
        self.mapped * self.page
    }

    /// Host bytes the pages with host memory of their own hold.
    pub(crate) fn host_mapped(&self) -> usize {
        self.hosted * self.page
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

/// No buffers: the holder of a room whose buffer its owner took out
/// ([`StackRoom`]).
impl Views for () {
    fn dissolve(self) {}
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
    /// each at a page boundary, built into `V` by `build`, their pages
    /// backed where `backing` says ([`Backing`]); `runs` are the rows of one buffer, which every buffer has at
    /// its own offset. Every page maps the spare: the rows a fit reaches get
    /// memory of their own, zeroed ([`Room::fit`]). `what` names the store
    /// the room backs, for a fit's named refusal. A room that can back
    /// pages on the host takes a page both locations can create
    /// ([`host::page_for`]) and is refused by name where the device cannot
    /// host a VMM page. Refused by name for a run past a buffer.
    pub(crate) fn with_backing<T, const N: usize>(
        stream: Arc<CudaStream>,
        len: usize,
        runs: &[Run],
        what: &'static str,
        backing: Backing,
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
        let page = match backing {
            Backing::Card => crate::allocation_granularity(ctx.ordinal())?,
            Backing::Host | Backing::Spill { .. } => host::page_for(ctx.ordinal())?,
        };
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
            backing,
            fitted: 0,
            reach: 0,
            mapped: 0,
            hosted: 0,
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

    /// The card-backed choice of [`Roomed::with_backing`], for the room tests.
    #[cfg(test)]
    pub(crate) fn new<T, const N: usize>(
        stream: Arc<CudaStream>,
        len: usize,
        runs: &[Run],
        what: &'static str,
        build: impl FnOnce([DeviceBuffer<T>; N]) -> V,
    ) -> Result<Roomed<V>, GpuError> {
        Roomed::with_backing(stream, len, runs, what, Backing::Card, build)
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

    /// The bytes of new pages a fit of `positions` positions still asks, over
    /// every room ([`Room::asks`]).
    pub(crate) fn asks(&self, positions: usize) -> usize {
        self.rooms.iter().map(|r| r.asks(positions)).sum()
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

    /// Card bytes the pages with card memory of their own hold, over every
    /// room ([`Room::mapped`]).
    pub(crate) fn mapped(&self) -> usize {
        self.rooms.iter().map(Room::mapped).sum()
    }

    /// Host bytes the pages with host memory of their own hold, over every
    /// room ([`Room::host_mapped`]).
    pub(crate) fn host_mapped(&self) -> usize {
        self.rooms.iter().map(Room::host_mapped).sum()
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

/// One routed expert stack held as a room: its buffer is reserved whole and
/// its pages backed at load, every one ([`StackRoom::new`]) or a prefix of them
/// ([`StackRoom::with_backed`]: the slots a plan filled, the rest reserved for a
/// fill), so with nothing released its address, length and bytes are a plain
/// buffer's. A residency machine gives back the pages wholly at or past a slot
/// of the stack ([`StackRoom::release_from`]) and backs them again
/// ([`StackRoom::restore`]); the range stays reserved, and the pages given back
/// or never backed map the device's spare page. The buffer
/// [`StackRoom::new`] returns is a view over the room's range: its holder
/// takes it apart without a free (`DeviceBuffer::into_raw_parts`) before the
/// room drops, as [`Views::dissolve`] does for a store's.
pub struct StackRoom {
    room: Mutex<Roomed<()>>,
    /// Device bytes of the pages given back and not backed again.
    released: AtomicU64,
}

impl StackRoom {
    /// A stack of `words` `u32` words on `stream`'s device, every page
    /// backed and zeroed on `stream`, and the buffer over its range. A
    /// residency machine's stacks are card memory always, so the pages'
    /// backing is [`Backing::Card`] by name, not by default. `what` names
    /// the stack for a refusal.
    pub fn new(
        stream: Arc<CudaStream>,
        words: usize,
        what: &'static str,
    ) -> Result<(DeviceBuffer<u32>, StackRoom), GpuError> {
        StackRoom::with_backed(stream, words, words, what)
    }

    /// [`StackRoom::new`] over `words` words with only the pages that hold a
    /// byte of the first `backed` words backed and zeroed: the rest of the
    /// range is reserved and maps the spare until a restore backs it
    /// ([`StackRoom::restore`], the machine's fill), and counts as released
    /// ([`StackRoom::released`]) from the start. `backed` past `words` is
    /// refused by name.
    pub fn with_backed(
        stream: Arc<CudaStream>,
        words: usize,
        backed: usize,
        what: &'static str,
    ) -> Result<(DeviceBuffer<u32>, StackRoom), GpuError> {
        if backed > words {
            return Err(GpuError::shape(
                WHAT,
                format!("{what}: {backed} words backed of a stack of {words}"),
            ));
        }
        let bytes = words
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| GpuError::shape(WHAT, format!("{words} words pass usize")))?;
        let whole = [Run {
            at: 0,
            row: bytes,
            per: 1,
            rows: 1,
        }];
        let mut roomed = Roomed::<DeviceBuffer<u32>>::with_backing(
            Arc::clone(&stream),
            words,
            &whole,
            what,
            Backing::Card,
            |[b]| b,
        )?;
        let unbacked = if backed == words {
            roomed.fill()?;
            0
        } else {
            let [room] = roomed.rooms_mut() else {
                return Err(GpuError::state(WHAT, "a stack room holds one room"));
            };
            room.restore(0, backed * size_of::<u32>(), &stream)?;
            room.whole().saturating_sub(room.mapped()) as u64
        };
        let mut view = None;
        let room = roomed.map(|b| view = Some(b));
        let view = view.ok_or_else(|| GpuError::state(WHAT, "the stack's buffer"))?;
        Ok((
            view,
            StackRoom {
                room: Mutex::new(room),
                released: AtomicU64::new(unbacked),
            },
        ))
    }

    /// A `rows × cols` tensor of words held as a room ([`StackRoom::with_backed`]
    /// over `rows · cols` words, the pages of the first `backed` rows backed: a
    /// stack of `rows` slots a plan filled `backed` of), the buffer owned by the
    /// tensor. The tensor is a view over the room's range; its holder takes it
    /// apart without a free (`DeviceTensor::into_buf`, then
    /// `DeviceBuffer::into_raw_parts`) before the room drops.
    pub(crate) fn tensor_backed(
        stream: Arc<CudaStream>,
        rows: usize,
        cols: usize,
        backed: usize,
        what: &'static str,
    ) -> Result<(DeviceTensor<u32>, StackRoom), GpuError> {
        let words = rows
            .checked_mul(cols)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{rows} × {cols} words pass usize")))?;
        let (buf, room) = StackRoom::with_backed(stream, words, backed * cols, what)?;
        // The buffer holds exactly `rows · cols` words, so the tensor takes it whole.
        let w = DeviceTensor::from_buf(buf, rows, cols)?;
        Ok((w, room))
    }

    /// `f` over the stack's one room, under the room's lock.
    fn with<R>(&self, f: impl FnOnce(&mut Room) -> Result<R, GpuError>) -> Result<R, GpuError> {
        let mut held = self.room.lock().unwrap_or_else(PoisonError::into_inner);
        match held.rooms_mut() {
            [room] => f(room),
            _ => Err(GpuError::state(WHAT, "a stack room holds one room")),
        }
    }

    /// The device bytes [`StackRoom::release_from`] `byte` would free: the
    /// pages wholly at or past `byte` that hold memory.
    pub fn tail_bytes(&self, byte: usize) -> Result<u64, GpuError> {
        self.with(|r| Ok(r.tail_bytes(byte)))
    }

    /// Every page wholly at or past byte `byte` given back to the spare, its
    /// memory freed: the device bytes freed ([`Room::release_from`], whose
    /// contract on the device's work the caller keeps).
    pub fn release_from(&self, byte: usize) -> Result<u64, GpuError> {
        self.with(|r| {
            let before = r.mapped();
            let freed = r.release_from(byte);
            self.released
                .fetch_add(before.saturating_sub(r.mapped()) as u64, Ordering::Relaxed);
            freed
        })
    }

    /// The device bytes [`StackRoom::restore`]`(from, to)` would back: the
    /// pages holding a byte of `[from, to)` that map the spare.
    pub fn unbacked(&self, from: usize, to: usize) -> Result<u64, GpuError> {
        self.with(|r| Ok(r.unbacked_bytes(from, to)))
    }

    /// Fresh memory, zeroed on `stream` in stream order, for every page that
    /// holds a byte of `[from, to)` and maps the spare: the device bytes
    /// backed ([`Room::restore`]).
    pub fn restore(&self, from: usize, to: usize, stream: &CudaStream) -> Result<u64, GpuError> {
        self.with(|r| {
            let before = r.mapped();
            let backed = r.restore(from, to, stream);
            self.released
                .fetch_sub(r.mapped().saturating_sub(before) as u64, Ordering::Relaxed);
            backed
        })
    }

    /// [`StackRoom::restore`] of every page from byte `from` on: the pages
    /// [`StackRoom::release_from`]`(from)` gave back.
    pub fn restore_from(&self, from: usize, stream: &CudaStream) -> Result<u64, GpuError> {
        self.restore(from, usize::MAX, stream)
    }

    /// Device bytes of the pages given back or never backed and not backed
    /// since.
    pub fn released(&self) -> u64 {
        self.released.load(Ordering::Relaxed)
    }

    /// Device bytes the pages that hold memory take.
    pub fn backed(&self) -> Result<usize, GpuError> {
        self.with(|r| Ok(r.mapped()))
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

    /// A stack room of 16 slots of 921,600 bytes, every page backed at the
    /// load, gives back exactly the pages wholly at or past slot 10's first
    /// byte: the count is derived here from the page and the slot, the
    /// device's free bytes grow by at least that many pages, the slots below
    /// read back their bytes, a second release frees nothing, and a restore
    /// backs the pages again zeroed with the slots below untouched. Runs
    /// alone (`hw_held_`): the free bytes are the device's, which a parallel
    /// test's allocation moves. FAIL-first: a release that frees nothing.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_stack_room_gives_back_the_pages_past_a_slot() {
        const PART: usize = 921_600;
        const SLOTS: usize = 16;
        const LINE: usize = 10;
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let pages = (SLOTS * PART).div_ceil(page);
        let keep = (LINE * PART).div_ceil(page);
        let gone = (pages - keep) * page;
        assert!(gone > 0, "the line leaves whole pages past it");
        assert!(
            12 * PART >= keep * page,
            "slot 12 lies wholly past the line"
        );
        let words = PART / size_of::<u32>();
        let (mut buf, room) =
            StackRoom::new(Arc::clone(&stream), SLOTS * words, "the tests' stack").expect("a room");
        let base = buf.cu_deviceptr();
        assert_eq!(room.backed().expect("backed"), pages * page, "every page");
        for s in 0..SLOTS {
            WindowMut::<u32>::of_mut(&mut buf, s * PART, words)
                .expect("a slot")
                .copy_from_host(&stream, &vec![s as u32 + 1; words])
                .expect("a write");
        }
        stream.synchronize().expect("the writes");
        let read = |buf: &DeviceBuffer<u32>, s: usize| {
            Window::<u32>::of(buf, s * PART, words)
                .expect("a slot")
                .to_host_vec(&stream)
                .expect("a read")
        };
        for s in 0..SLOTS {
            assert_eq!(read(&buf, s), vec![s as u32 + 1; words], "slot {s} before");
        }
        assert_eq!(
            room.tail_bytes(LINE * PART).expect("the tail"),
            gone as u64,
            "the tail the release would free"
        );
        ctx.bind_to_thread().expect("the context binds");
        let (before, _) = crate::current_mem_info().expect("the device's free bytes");
        let freed = room.release_from(LINE * PART).expect("a release");
        assert_eq!(freed, gone as u64, "the pages wholly past slot {LINE}");
        let (after, _) = crate::current_mem_info().expect("the device's free bytes");
        assert!(
            after >= before + gone,
            "the device's free bytes grew by the pages: {before} then {after}, {gone} given back"
        );
        assert_eq!(room.backed().expect("backed"), keep * page);
        assert_eq!(room.released(), gone as u64);
        for s in 0..LINE {
            assert_eq!(read(&buf, s), vec![s as u32 + 1; words], "slot {s} after");
        }
        assert_eq!(
            room.release_from(LINE * PART).expect("a second release"),
            0,
            "pages already given back free nothing"
        );
        assert_eq!(room.released(), gone as u64);
        assert_eq!(buf.cu_deviceptr(), base, "the address stays");
        let backed = room.restore_from(LINE * PART, &stream).expect("a restore");
        assert_eq!(backed, gone as u64, "the pages given back backed again");
        assert_eq!(room.backed().expect("backed"), pages * page);
        assert_eq!(room.released(), 0);
        for s in 0..LINE {
            assert_eq!(
                read(&buf, s),
                vec![s as u32 + 1; words],
                "slot {s} restored"
            );
        }
        for s in [12, SLOTS - 1] {
            assert_eq!(read(&buf, s), vec![0u32; words], "slot {s} reads zero");
        }
        stream.synchronize().expect("the room's work finished");
        // The view is taken apart without a free before the room drops.
        drop(buf.into_raw_parts());
        drop(room);
    }

    /// A `Backing::Host` room of 4 pages fitted whole leaves the device's
    /// free bytes within one page of what they were (a room's only card
    /// memory is its spare, made at construction): the pages' memory sits
    /// on the host, `host_mapped()` says the 4 pages and `mapped()` none; a
    /// fresh fit's pages read back zeros and bytes written through the
    /// buffer read back equal; a settle after a reach of one page leaves
    /// one page of host memory; a fit of the pages that settle gave back
    /// reads zeros again — the pages last held a written pattern, so a page
    /// not zeroed reads it. Runs alone (`hw_held_`): the free bytes are the
    /// device's, which a parallel test's allocation moves.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_host_room_backs_its_pages_on_the_host() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        ctx.bind_to_thread().expect("the context binds");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, rows) = (page / 2, 8);
        let words = row / size_of::<u32>();
        let runs = [Run {
            at: 0,
            row,
            per: 1,
            rows,
        }];
        let mut r: Roomed<[DeviceBuffer<u32>; 1]> = Roomed::with_backing(
            Arc::clone(&stream),
            rows * words,
            &runs,
            "the tests' host planes",
            Backing::Host,
            |b| b,
        )
        .expect("a host room");
        let free = || {
            crate::capsync::ctx_sync(&ctx, "the host room's check").expect("the context syncs");
            crate::current_mem_info()
                .expect("the device's free bytes")
                .0
        };
        let before = free();
        r.fit(rows).expect("a fit of every row");
        let after = free();
        assert!(
            before.saturating_sub(after) <= page,
            "a host fit takes no card bytes beyond the spare: {before} then {after}, page {page}"
        );
        assert_eq!(r.host_mapped(), 4 * page, "every page on the host");
        assert_eq!(r.mapped(), 0, "no page on the card");
        let read = |r: &Roomed<[DeviceBuffer<u32>; 1]>, j: usize| {
            Window::<u32>::of(&r[0], j * row, words)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")
        };
        let write = |r: &mut Roomed<[DeviceBuffer<u32>; 1]>, j: usize, v: u32| {
            WindowMut::<u32>::of_mut(&mut r[0], j * row, words)
                .expect("a backed row")
                .copy_from_host(&stream, &vec![v; words])
                .expect("a write");
        };
        for j in 0..rows {
            assert_eq!(read(&r, j), vec![0u32; words], "row {j} of a fresh fit");
        }
        for j in 0..rows {
            write(&mut r, j, j as u32 + 1);
        }
        for j in 0..rows {
            assert_eq!(read(&r, j), vec![j as u32 + 1; words], "row {j} reads back");
        }
        r.settle().expect("a settle");
        assert_eq!(
            r.host_mapped(),
            4 * page,
            "a reach of every row keeps every page"
        );
        r.fit(1).expect("a fit");
        r.settle().expect("a settle");
        assert_eq!(
            r.host_mapped(),
            page,
            "a reach of one row keeps the first page alone"
        );
        r.fit(rows).expect("a fit of every row again");
        for j in 0..2 {
            assert_eq!(
                read(&r, j),
                vec![j as u32 + 1; words],
                "the kept page keeps its bytes"
            );
        }
        for j in 2..rows {
            assert_eq!(
                read(&r, j),
                vec![0u32; words],
                "row {j} of a page backed again reads zero"
            );
        }
        stream.synchronize().expect("the room's work finished");
    }

    /// A `Backing::Spill` room whose fit runs with the card squeezed to
    /// under a page free backs its pages on the host: the fit completes —
    /// the card cannot make a page's memory — `host_mapped()` holds pages,
    /// and bytes written through the buffer read back equal. The squeezing
    /// allocations are freed after. Runs alone (`hw_held_`): the squeeze
    /// holds the device's free bytes, which a parallel test's allocation
    /// would shift. FAIL-first: a spill that hands the card's refusal back.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_spill_room_backs_on_the_host_when_the_card_is_full() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        ctx.bind_to_thread().expect("the context binds");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, rows) = (page / 2, 4);
        let words = row / size_of::<u32>();
        let runs = [Run {
            at: 0,
            row,
            per: 1,
            rows,
        }];
        let mut r: Roomed<[DeviceBuffer<u32>; 1]> = Roomed::with_backing(
            Arc::clone(&stream),
            rows * words,
            &runs,
            "the tests' spill planes",
            Backing::Spill { margin: 0 },
            |b| b,
        )
        .expect("a spill room");
        let (free, _) = crate::current_mem_info().expect("the device's free bytes");
        let bulk = DeviceBuffer::<u8>::zeroed(&stream, free.saturating_sub(8 * page) & !(page - 1))
            .expect("the device's free bytes less a few pages");
        let mut squeeze = Vec::new();
        while crate::current_mem_info()
            .expect("the device's free bytes")
            .0
            >= page
        {
            match PhysicalAllocation::new(ctx.cu_device(), page) {
                Ok(card) => squeeze.push(card),
                Err(_) => break,
            }
        }
        r.fit(rows)
            .expect("a fit past the card's bytes lands on the host");
        assert!(r.host_mapped() > 0, "the spill backed pages on the host");
        let read = |r: &Roomed<[DeviceBuffer<u32>; 1]>, j: usize| {
            Window::<u32>::of(&r[0], j * row, words)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")
        };
        for j in 0..rows {
            WindowMut::<u32>::of_mut(&mut r[0], j * row, words)
                .expect("a backed row")
                .copy_from_host(&stream, &vec![j as u32 + 7; words])
                .expect("a write");
        }
        for j in 0..rows {
            assert_eq!(read(&r, j), vec![j as u32 + 7; words], "row {j} reads back");
        }
        drop(squeeze);
        drop(bulk);
        stream.synchronize().expect("the room's work finished");
    }

    /// A `Backing::Spill` room of twelve pages with a margin of six pages
    /// under the card's free bytes keeps that margin: after the fit of every
    /// row the device's free bytes are at least the margin, the pages past
    /// the margin sit on the host (`host_mapped()` holds pages, `mapped()`
    /// no more than the six the margin leaves) and every page is backed.
    /// Runs alone (`hw_held_`): the free bytes are the device's, which a
    /// parallel test's allocation moves. FAIL-first: a spill that waits for
    /// the driver's refusal takes all twelve pages from the card.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_spill_room_keeps_the_cards_margin() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        ctx.bind_to_thread().expect("the context binds");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, rows) = (page, 12);
        let words = row / size_of::<u32>();
        let runs = [Run {
            at: 0,
            row,
            per: 1,
            rows,
        }];
        let free = || {
            crate::capsync::ctx_sync(&ctx, "the spill room's check").expect("the context syncs");
            crate::current_mem_info()
                .expect("the device's free bytes")
                .0
        };
        // Seven pages over the margin: the room's spare page (made at
        // construction) and six pages of the room's own.
        let margin = free().saturating_sub(7 * page);
        let mut r: Roomed<[DeviceBuffer<u32>; 1]> = Roomed::with_backing(
            Arc::clone(&stream),
            rows * words,
            &runs,
            "the tests' spill planes",
            Backing::Spill {
                margin: margin as u64,
            },
            |b| b,
        )
        .expect("a spill room with a margin");
        assert_eq!((r.mapped(), r.host_mapped()), (0, 0), "no page backed yet");
        r.fit(rows).expect("a fit of every row");
        let after = free();
        assert!(
            after >= margin,
            "the fit left {after} B free, under the margin of {margin} B"
        );
        assert!(
            r.host_mapped() > 0,
            "the pages past the margin are on the host"
        );
        assert!(
            r.mapped() <= 6 * page,
            "the card holds {} B of pages, the margin leaves room for six",
            r.mapped()
        );
        assert_eq!(
            r.mapped() + r.host_mapped(),
            rows * page,
            "every page is backed, on the card or the host"
        );
        stream.synchronize().expect("the room's work finished");
    }

    /// Rooms share pages across backings, both ways: a card room's prefix
    /// shared into a host room, then the host room shared into a card room
    /// — the layout one buffer of four rows, two a page, so a share of three
    /// positions maps the source's first page (whole rows of the prefix) and
    /// copies the second (where the prefix ends); after
    /// `own_from(1)` the destination's rows still equal the source's (the
    /// page took its bytes with it), and a write to the destination leaves
    /// the source's row as it was. Runs alone with the held room tests
    /// (`hw_held_`).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_rooms_share_across_backings() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, rows) = (page / 2, 4);
        let words = row / size_of::<u32>();
        let runs = [Run {
            at: 0,
            row,
            per: 1,
            rows,
        }];
        let room = |backing: Backing| -> Roomed<[DeviceBuffer<u32>; 1]> {
            Roomed::with_backing(
                Arc::clone(&stream),
                rows * words,
                &runs,
                "the tests' shared planes",
                backing,
                |b| b,
            )
            .expect("a room")
        };
        let read = |r: &Roomed<[DeviceBuffer<u32>; 1]>, j: usize| {
            Window::<u32>::of(&r[0], j * row, words)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")
        };
        let write = |r: &mut Roomed<[DeviceBuffer<u32>; 1]>, j: usize, v: u32| {
            WindowMut::<u32>::of_mut(&mut r[0], j * row, words)
                .expect("a backed row")
                .copy_from_host(&stream, &vec![v; words])
                .expect("a write");
        };
        let source = |backing: Backing| -> Roomed<[DeviceBuffer<u32>; 1]> {
            let mut s = room(backing);
            s.fit(rows).expect("a fit");
            for j in 0..rows {
                write(&mut s, j, j as u32 + 1);
            }
            s
        };
        // A card room's pages into a host room, then the other way.
        for (side, other) in [
            (Backing::Card, Backing::Host),
            (Backing::Host, Backing::Card),
        ] {
            let a = source(side);
            let mut b = room(other);
            assert_eq!(
                b.share(&a, 3).expect("a share across backings"),
                SharedPages {
                    mapped: page,
                    copied: page
                },
                "{side:?} into {other:?}: the first page mapped, the second copied"
            );
            for j in 0..rows {
                assert_eq!(read(&b, j), read(&a, j), "{other:?} row {j} of the share");
            }
            b.own_from(1).expect("an own from one");
            for j in 0..rows {
                assert_eq!(
                    read(&b, j),
                    read(&a, j),
                    "{other:?} row {j} after the page took its own memory"
                );
            }
            write(&mut b, 0, 999);
            assert_eq!(
                read(&b, 0),
                vec![999; words],
                "{other:?} row 0 after the write"
            );
            assert_eq!(
                read(&a, 0),
                vec![1; words],
                "{side:?} row 0 stands after the write to {other:?}"
            );
            b.settle().expect("a settle");
            for j in 0..rows {
                assert_eq!(
                    read(&a, j),
                    vec![j as u32 + 1; words],
                    "{side:?} row {j} stands"
                );
            }
        }
        stream.synchronize().expect("the rooms' work finished");
    }
}
