//! The NVMe expert tier's RAM arena: one anonymous mapping
//! ([`engram::direct::AnonMap`]), or the pages a [`PageSource`] provides
//! ([`ArenaKind::Provided`]; on the card, pinned host memory it maps),
//! carved into slots — one expert a slot,
//! each of its parts in its own [`DIRECT_ALIGN`]-aligned window, the bytes
//! landing at the skew an aligned read of the part's file range needs; a
//! matrix a checkpoint folder keeps as two planes, each in any shard, reads
//! its codes and its block scales into a window each, and the fill moves the
//! scales to follow the codes, the one run the host's planes read takes —
//! filled on a miss by [`DirectFile::read_span_uninit`] — over the resident pool for
//! a decode step's misses, on the tier's own reader for a prompt batch's and
//! a residency flip's victim — and lent to the host leg through the model
//! side's [`TierSlots`] seam or, on an arena whose pages have a device
//! address, held for the step port's card leg, which computes a step's ids
//! from their slots on the card
//! ([`NvTier::issue_card_hold`], then [`CardTicket::finish`]).
//! The file mapping serves the plan's host segment; every other id the leg
//! reads — the NVMe segment's, and the card's victims once they flip — is
//! this arena's, its slot refused by name while unfilled.
//!
//! The arena's own pages leave only when a landing frees a slot
//! ([`NvTier::release`]), `MADV_DONTNEED` on the slot's windows through the
//! arena's one give-back (`ArenaMem::give_back`, the only door to the
//! mapping's advice), which frees an anonymous range's pages — that is why
//! the arena is anonymous and not the page cache, whose warm set nothing
//! else can flush. A provided arena returns none: its allocation is the
//! plan's for the load. On write-combined pages (an integrated device's) a
//! `MADV_DONTNEED` leaves the range unbacked for the host and for a
//! `cuMemcpyDtoH` from its device address: the next fill's read fails with
//! `EFAULT`, and a CPU read or that copy faults. On an RTX 5090 the
//! allocation without write-combining (`PORTABLE | DEVICEMAP`, a discrete
//! card's) is a shared mapping, which keeps its pages through the advice. An
//! eviction returns none: the slot's next fill, in the same call, overwrites
//! the bytes it reads in place. The model
//! file's mapping is its readers': a lane's copy of an id the arena holds
//! unfilled reads it through the mapping, which faults its pages in from the
//! drive, and once the copy has consumed them the tier drops them from the
//! mapping and the page cache — the layer's every id but the ones a lane has
//! open ([`NvTier::end_read`], so the kernel's readahead past the copied id
//! goes too) — so nothing a lane brought in stays past it. A prompt call's
//! union and a decode step read the arena's ids from their slots
//! ([`NvTier::prefetch_pinned`] and [`NvTier::ensure_pinned`]; the step
//! port, `HostLayer::experts_step_union_into`), so their reads bring in no
//! page of the mapping to drop: a step's ids are the decode's hot set, which
//! the next step reads again from the arena. The lane only names the pages;
//! the tier's dropper thread makes the syscalls, a layer's request replacing
//! one still queued, so the freeing of the pages runs beside the next
//! layer's work rather than on the lane's thread ([`NvTier::flush`] waits
//! for the queue). Which pages go is the books' ([`drop_runs`]), never
//! `mincore`'s, and no page that holds a byte of the plan's host segment, of
//! an id a lane has open, or of the tensors beside a stack is named. The
//! books are one atomic hint an id and a per-slot state word a pick reads
//! without the lock; the LRU and the fills' slot ownership live under one
//! lock taken per `ensure`, never per pick. The engine serves a model's host
//! leg one service at a time, so no pick reads a slot another service
//! evicts — the seat and the run-ahead that share the arena across threads
//! join their fills before the books give the slot away.
//!
//! A lane's copy of an id the arena holds filled reads the slot, not the
//! mapping: [`NvTier::open_read`] pins the slot (the LRU takes no pinned
//! slot, and a miss that finds every candidate pinned is refused by name),
//! [`NvTier::staged_slot`] lends its bytes and [`NvTier::end_read`] lets it
//! go and queues no drop, since the copy read no page of the model file's
//! mapping. An id the arena holds unfilled when the lane opens it reads the
//! mapping and drops as before. An id on the card holds no slot: the machine
//! frees the promoted id's slot at the landing ([`NvTier::release`], through
//! `SwapSource::release_host`), so every demotion fills. A victim's fill runs
//! on the tier's fill reader at the victim rank, behind every queued prompt
//! batch ([`NvTier::ensure_victim`]), and never on the pool, whose one
//! dispatch mutex is the chain's. Claims, pins and publishes take the books'
//! lock; a lane's read entry also takes the drop queue lock, so a page
//! release cannot overlap it.
//!
//! A call's claim kind ([`Claim`]) places what it takes in the books. A
//! step's and a keep's slots take a fresh tick; a transient's enter at the
//! cold end (tick 0 and a sequence number in claim order), so the next miss
//! takes the oldest transient before any hot slot, and a later step or keep
//! hit turns a transient hot; a transient claim's hit leaves a hot or
//! transient slot as it is. Among the hot slots a miss takes the least
//! recently used under `BLOOMERY_NVTIER_EVICT=tick` (unset), and under `heat`
//! the one whose expert has the least decayed routing count ([`LayerCounts`],
//! the least recently used among equal counts). A call's own ids are never
//! its victims under either policy. A batch pin ([`NvTier::pin`]) counts the ids
//! of a batch in the same open-read counts a lane's read does, so the books
//! take none of them until the guard drops. A prompt call's union claims
//! each batch of its experts as one claim and pin ([`NvTier::prefetch_pinned`]):
//! a hit counts at the claim and a miss as its fill publishes, each under the
//! one hold of the lock the claim or the publish takes, so no claim between
//! them takes a slot of the batch.
//!
//! A prompt batch's fill runs off the caller, on the tier's own reader:
//! [`NVTIER_FILL_THREADS`] threads started at the build, each blocked in one
//! `O_DIRECT` read at a time, taking one queue by rank with a prompt batch's
//! picks first and a flip's victim behind them. A unit is one part of a pick,
//! so a fill's three parts are three reads in flight, and a pick is published
//! only once all three have landed. The claim and the pin stay on
//! the caller; [`Ticket::wait`] joins the fill and publishes the picks under
//! the books' lock. A pick's reads write straight into its slot's windows,
//! over the expert it evicted, and the threads never take the pool's dispatch
//! lock or the books' lock, and never wait on the card.

use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bloomery_levers::NvTierEvict;
use engram::direct::{AnonMap, DIRECT_ALIGN, DirectFile, aligned_span};
use gguf::Split;
use model::moe::{Claim, PinRelease, PinToken, Ticket, TicketWait, TierError, TierPin, TierSlots};
use model::placement::host_lock::{HostFile, HostSet, PageDrop, expert_runs, page_bytes};
use model::placement::paged_drop::{Mark, drop_runs, overlap_of};
use model::placement::workstation::ArenaGrant;
use model::placement::{
    ARENA_ALIGN, ArenaFloor, Device, ModelTensor, Plan, Role, arena_part_bytes,
};
use model::r8file::R8Source;

use super::arena_leg::LegOff;
use crate::GpuError;

const _: () = assert!(model::placement::STEP_PORT_COLUMNS == model::ops::DEFER_MAX_COLS);

mod reader;
use reader::{Job, Kind, Reader};
pub use reader::{KindStats, ReaderStats};

/// Threads of the tier's fill reader: the depth the drive is read at. The
/// drive's rate stops rising near it; a deeper queue puts more reads in
/// flight and no more bytes through.
pub(crate) const NVTIER_FILL_THREADS: usize = 16;

/// How long the tier's drop waits for the reader's threads to leave their
/// reads.
const READER_STOP_BOUND: Duration = Duration::from_secs(30);

/// The arena's one mapping, and the one door its pages and bytes pass
/// through. The mapping is a private field of [`ArenaMem`]: a slot's pages go
/// back through [`ArenaMem::give_back`] and no other call; and a fill writes
/// straight into a slot's windows ([`ArenaMem::read_into`]), never through a
/// bounce buffer. The mapping is anonymous memory, or pages a source
/// provides ([`ArenaKind`]), which stay for the arena's life.
mod arena_mem {
    use engram::EngramError;
    use engram::direct::{AnonMap, DIRECT_ALIGN, DirectFile};

    use super::ArenaPages;

    enum Pages {
        Anon(AnonMap),
        Provided(Box<dyn ArenaPages>),
    }

    pub(super) struct ArenaMem {
        pages: Pages,
        /// The mapping's first byte, as the fills write it.
        base: *mut u8,
        len: usize,
    }

    // SAFETY: `base` points inside the allocation `pages` owns, which lives
    // and moves with this value.
    unsafe impl Send for ArenaMem {}
    // SAFETY: the mapping is read only through `slice`, over a slot a pin
    // or a lane's open read holds filled ([`ArenaMem::slice`]'s contract),
    // and written only through `read_into`, whose callers own disjoint
    // windows (a slot's, by the books): no claim takes a pinned slot, so no
    // byte is read and written at once. The card's alias of pinned pages
    // reads no window either: the only device address a slot gives is a
    // pin's ([`model::moe::TierPin::device_slot`]), under the same contract.
    unsafe impl Sync for ArenaMem {}

    impl ArenaMem {
        pub(super) fn new(map: AnonMap) -> ArenaMem {
            ArenaMem {
                base: map.as_ptr().cast_mut(),
                len: map.bytes().len(),
                pages: Pages::Anon(map),
            }
        }

        /// The `len` bytes `pages` hold, from the first byte they name,
        /// which a direct fill's window starts on. A source that gives
        /// another length, or a first byte off a [`DIRECT_ALIGN`] boundary,
        /// breaks its contract ([`ArenaPages`]): refused by name.
        pub(super) fn provided(pages: Box<dyn ArenaPages>, len: usize) -> ArenaMem {
            let base = pages.base();
            assert!(
                pages.bytes() == len,
                "the page source gave {} B for the arena's {len} B",
                pages.bytes()
            );
            assert!(
                (base as usize).is_multiple_of(DIRECT_ALIGN),
                "the provided arena's first byte {base:p} is not {DIRECT_ALIGN}-aligned"
            );
            ArenaMem {
                base,
                len,
                pages: Pages::Provided(pages),
            }
        }

        /// The mapping's first byte's address.
        pub(super) fn start(&self) -> usize {
            self.base as usize
        }

        /// The mapping's first byte as the card addresses it; `None` on an
        /// anonymous mapping and on pages with no device address.
        pub(super) fn device_start(&self) -> Option<u64> {
            match &self.pages {
                Pages::Anon(_) => None,
                Pages::Provided(pages) => pages.device_base(),
            }
        }

        /// The mapping's length.
        pub(super) fn len(&self) -> usize {
            self.len
        }

        /// The mapping's byte `at`, the first of `len` inside it. Panics
        /// past the mapping.
        fn window(&self, at: usize, len: usize) -> *mut u8 {
            assert!(
                at.checked_add(len).is_some_and(|end| end <= self.len),
                "the arena's window {at}+{len} is past its {} B",
                self.len
            );
            self.base.wrapping_add(at)
        }

        /// Bytes `at .. at + len` of the mapping, a slot's part. The caller
        /// holds the slot filled through a pin or a lane's open read while
        /// the borrow lives ([`model::moe::TierSlots::slot`]'s contract): no
        /// claim takes such a slot, so no fill writes it, and its fill wrote
        /// every byte of the part (a fill whose read placed fewer than the
        /// part's bytes is refused, and its slot never reads filled). Panics
        /// past the mapping.
        pub(super) fn slice(&self, at: usize, len: usize) -> &[u8] {
            // SAFETY: inside the mapping (`window` checks), which lives as
            // long as `self`; by this function's contract the fill wrote the
            // bytes and no fill writes them while the borrow lives.
            unsafe { std::slice::from_raw_parts(self.window(at, len), len) }
        }

        /// Read `len` bytes of `file` from `from` into the mapping at `at`,
        /// the bytes placed (the file's end may cut them short): those bytes
        /// are the file's, and the rest of the window keeps what it held,
        /// which on pinned pages may be bytes never written. The window
        /// is the caller's to own — a slot's, which the books gave this
        /// pick alone — until the read returns: no other thread reads or
        /// writes any of it. That bound is kept in the books, not in the
        /// types, and the function is private to the tier. Panics past the
        /// mapping.
        pub(super) fn read_into(
            &self,
            file: &DirectFile,
            from: u64,
            at: usize,
            len: usize,
        ) -> Result<usize, EngramError> {
            self.with_window(at, len, |window| file.read_span_uninit(window, from))
        }

        /// Move `len` bytes of the mapping from `from` to `to`, the two
        /// ranges free to overlap: a fill's join of a part's runs, once its
        /// reads have landed, inside the slot the books gave its pick alone
        /// ([`ArenaMem::read_into`]'s bound). Panics past the mapping.
        pub(super) fn move_within(&self, from: usize, to: usize, len: usize) {
            let lo = from.min(to);
            let span = from.max(to) + len - lo;
            self.with_window(lo, span, |window| {
                window.copy_within(from - lo..from - lo + len, to - lo);
            });
        }

        /// `write` over bytes `at .. at + len` of the mapping, viewed as bytes
        /// that may be unwritten: the window of a slot the books gave the
        /// calling pick alone, which no other thread reads or writes until
        /// the fill publishes it. Panics past the mapping.
        fn with_window<R>(
            &self,
            at: usize,
            len: usize,
            write: impl FnOnce(&mut [std::mem::MaybeUninit<u8>]) -> R,
        ) -> R {
            // SAFETY: inside the mapping (`window` checks), viewed as bytes
            // that may be unwritten; the window is the calling pick's own
            // slot's, owned through the books until the fill publishes it,
            // and the picks of a fill are disjoint slots.
            let window = unsafe {
                std::slice::from_raw_parts_mut(
                    self.window(at, len).cast::<std::mem::MaybeUninit<u8>>(),
                    len,
                )
            };
            write(window)
        }

        /// Return the pages of `at .. at + len` — whole pages of a slot the
        /// books gave up — to the kernel: the next touch reads zeroes. The
        /// one place the arena's pages are advised. Nothing on provided
        /// pages: the allocation is the plan's for the load, and on
        /// write-combined pages advice leaves the range unbacked for the next
        /// fill and for a copy from its device address (the module's head).
        pub(super) fn give_back(&self, at: usize, len: usize) -> Result<(), EngramError> {
            match &self.pages {
                Pages::Anon(map) => map.dontneed(at, len),
                Pages::Provided(_) => Ok(()),
            }
        }
    }
}
use arena_mem::ArenaMem;

/// The memory behind a provided arena: [`ArenaPages::bytes`] bytes from a
/// first byte aligned to [`DIRECT_ALIGN`] (the direct fills' windows start
/// on it), alive as long as the value.
pub trait ArenaPages: Send + Sync {
    /// The first byte the fills write, the same on every call.
    fn base(&self) -> *mut u8;
    /// The mapping's length in bytes.
    fn bytes(&self) -> usize;
    /// The first byte as the card addresses it; `None` when the pages have
    /// no device address.
    fn device_base(&self) -> Option<u64>;
}

/// What makes the pages of a provided arena, once the carve knows how many
/// it needs.
pub trait PageSource {
    /// `len` bytes of pages ([`ArenaPages::bytes`] is `len`); refused by name
    /// when the source has none to give.
    fn alloc(&self, len: usize) -> Result<Box<dyn ArenaPages>, GpuError>;
}

/// The memory an arena is ([`NvTier::of_paged`]): an anonymous mapping, or
/// the pages a [`PageSource`] provides, whose filled slots a pin gives the
/// card's address of ([`TierPin::device_slot`]) when the pages have one.
#[derive(Clone, Copy)]
pub enum ArenaKind<'a> {
    Anon,
    Provided(&'a dyn PageSource),
}

/// A slot's state, as a pick reads it without the lock.
const FREE: u32 = 0;
const FILLING: u32 = 1;
const FILLED: u32 = 2;

/// The most parts of an expert the arena holds, a gated expert's gate, up
/// and down; a relu² expert holds two, its up and its down
/// ([`LayerArena::rows`]). A slot's windows lie in the plan's stack-row
/// order, and [`TierSlots::slot`]'s `part` names the leg's three, `[gate,
/// up, down]` — mapped to the layer's rows at the attach
/// ([`NvTier::note_parts`]), a relu² leg's gate to its up. A pick is that
/// many units of a fill, one a part; a part the layer does not hold reads
/// nothing.
const PARTS: usize = 3;

/// One run of a part's bytes: `len` bytes at file offset `at` of the split's
/// shard `shard`.
#[derive(Clone, Copy, Default)]
struct Run {
    shard: usize,
    at: u64,
    len: usize,
}

/// Each of the `n` experts' runs of routed stack `t` in `split`, `(codes, scales)`, where the
/// host reads them ([`expert_runs`]): a stack a checkpoint folder locates
/// ([`gguf::Place::Stack`]) reads each matrix where the folder keeps it, its codes and, kept as
/// two planes, its block scales, each in its own shard; every other stack is one run of its file
/// its experts cut evenly. Refused by name besides the runs' own refusals (a stack the split does
/// not hold, an expert count other than `n`, bytes the experts do not cut evenly): a stack no
/// file holds (a made one) and a located stack of another matrix count.
fn stack_runs(split: &Split, t: &ModelTensor, n: u32) -> Result<Vec<(Run, Option<Run>)>, GpuError> {
    const WHAT: &str = "NvTier::of_paged";
    let name = &t.name;
    match split.place(name) {
        Some(gguf::Place::Made) => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{name}: no file holds it, and the NVMe arena reads a stack where a file keeps it"
                ),
            ));
        }
        Some(gguf::Place::Stack { codes_at, .. }) if codes_at.len() != n as usize => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{name}: {} matrices, the layer has {n} experts",
                    codes_at.len()
                ),
            ));
        }
        _ => {}
    }
    let run = |(file, at): (HostFile, Range<u64>)| match file {
        HostFile::Shard(shard) => Ok(Run {
            shard,
            at: at.start,
            len: usize::try_from(at.end - at.start).map_err(|_| {
                GpuError::shape(WHAT, format!("{name}: one expert's bytes pass usize"))
            })?,
        }),
        HostFile::Sidecar(_) => Err(GpuError::shape(
            WHAT,
            format!("{name}: an expert run in a sidecar"),
        )),
    };
    (0..n)
        .map(|id| {
            let [codes, scales] = expert_runs(R8Source::rows(split), t, u64::from(n), id, false)
                .map_err(|e| GpuError::plan(WHAT, e))?;
            let scales = if scales.1.is_empty() {
                None
            } else {
                Some(run(scales)?)
            };
            Ok((run(codes)?, scales))
        })
        .collect()
}

/// Open the fills' handle of shard `shard` into `files` (by shard) unless it
/// is there, `O_DIRECT` (`direct`) or through the page cache, for a run of
/// stack `name`. Refused by name: a shard the split does not hold or has no
/// path for, and one that is an anonymous copy, not its file's mapping
/// ([`gguf::Gguf::file_backed`]): the tier drops a paged stack's pages after
/// each read, which would zero the copy.
fn open_shard(
    split: &Split,
    files: &mut Vec<Option<Arc<DirectFile>>>,
    shard: usize,
    direct: bool,
    name: &str,
) -> Result<(), GpuError> {
    const WHAT: &str = "NvTier::of_paged";
    if files.get(shard).is_some_and(Option::is_some) {
        return Ok(());
    }
    let gguf = split.shard(shard).ok_or(GpuError::shape(
        WHAT,
        format!("{name}: shard {shard} is not there"),
    ))?;
    if !gguf.file_backed() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name}: shard {shard} is an anonymous copy, not its file's mapping: the tier \
                 drops a paged stack's pages after each read, which would zero the copy"
            ),
        ));
    }
    let path = split.shard_path(shard).ok_or(GpuError::shape(
        WHAT,
        format!("{name}: shard {shard} has no path"),
    ))?;
    let file = if direct {
        DirectFile::open(path)
    } else {
        DirectFile::open_buffered(path)
    }
    .map_err(|e| GpuError::plan(WHAT, e))?;
    if files.len() <= shard {
        files.resize(shard + 1, None);
    }
    files[shard] = Some(Arc::new(file));
    Ok(())
}

/// One part of one expert as its files hold it — static per (layer, id,
/// part), resolved at the build: its one run (`codes`) or, for a matrix a
/// checkpoint folder keeps as two planes, its codes' run and its block
/// scales' run (`scales`), each in its own shard; and the skew the codes'
/// aligned read lands at in the part's window. A slot lends the codes, then
/// the scales ([`PartRange::len`] bytes). The default holds no run, the
/// part a layer of fewer rows than [`PARTS`] does not have.
#[derive(Clone, Copy, Default)]
struct PartRange {
    codes: Run,
    scales: Option<Run>,
    skew: usize,
}

impl PartRange {
    /// The part's bytes as its slot lends them.
    fn len(&self) -> usize {
        self.codes.len + self.scales.map_or(0, |s| s.len)
    }

    /// The part's runs: the codes, then the scales.
    fn runs(&self) -> impl Iterator<Item = Run> {
        std::iter::once(self.codes).chain(self.scales)
    }
}

// The placement's slot geometry is the arena's: its windows align to the
// pages the direct reads land in.
const _: () = assert!(ARENA_ALIGN as usize == DIRECT_ALIGN);

/// The bytes a part's windows take in its slot: a window for each run its
/// fill reads, the run's own bytes and the two partial pages an aligned read
/// of them can span — the placement's one shape of a part
/// ([`arena_part_bytes`]). `scales` is 0 for a part of one run.
fn window_bytes(codes: usize, scales: usize) -> usize {
    arena_part_bytes(codes as u64, scales as u64) as usize
}

/// The slots a layer holds for the arena's `budget` bytes over the paged
/// layers' slot sizes: the same count a layer, so every layer warms alike
/// and the carved total stays under the budget. A count under the arena's
/// floor (`least`, the placement's [`ArenaFloor`]: one call's distinct ids
/// and the lane's slots beside them, or a prompt union's two batches and the
/// lane's slots), or under one slot, is the named refusal — a call would
/// evict its own ids, or every read refuse.
pub(crate) fn slots_of(
    budget: u64,
    slot_bytes: &[usize],
    least: &ArenaFloor,
) -> Result<usize, String> {
    let total: usize = slot_bytes.iter().sum();
    if total == 0 {
        return Err("the plan pages no routed stack".to_string());
    }
    let n = (budget as usize) / total;
    let want = least.slots.max(1);
    if (n as u64) < want {
        return Err(format!(
            "the arena's budget {budget} B carves {n} slots a paged layer over the paged layers' \
             {total} B a slot set, under the floor of {want} slots (the larger of {} × {} ids a \
             call + {} lane slots and {} for a union in batches, {} B over these layers): a \
             call would evict its own ids",
            least.k,
            least.columns,
            least.lane_slots,
            least.union,
            want.saturating_mul(total as u64)
        ));
    }
    Ok(n)
}

/// Why a miss found no slot to take.
#[derive(Debug, PartialEq, Eq)]
enum NoSlot {
    /// Every slot is filling: two ensures raced the layer.
    Filling,
    /// No slot is free and every filled one is pinned by a lane's open read
    /// or a batch pin, or is one of the call's own ids: the arena's floor does not
    /// cover the pins open beside the call.
    Pinned,
}

/// The slot a miss takes: the first free one, else the victim of the filled
/// ones `pinned` does not name — the oldest transient (the least `seq`: a
/// transient sits at tick 0), else the least recently used, or with the
/// layer's `counts` ([`NvTierEvict::Heat`]) the hot slot whose expert has the
/// least count, the least recently used among equal counts. A filling slot is
/// never the arena's to take, its fill owns it, and a pinned one is being
/// read. Refused by name when none is left, never waited for and never
/// overwritten: all filling is the race of two ensures, all the rest pinned
/// the floor's.
fn pick_slot(
    slots: &[SlotBook],
    pinned: impl Fn(&SlotBook) -> bool,
    counts: Option<&LayerCounts>,
) -> Result<usize, NoSlot> {
    if let Some(k) = slots
        .iter()
        .position(|s| s.state.load(Ordering::Relaxed) == FREE)
    {
        return Ok(k);
    }
    let filled = || {
        slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state.load(Ordering::Relaxed) == FILLED)
    };
    let tick = |s: &SlotBook| {
        (
            s.tick.load(Ordering::Relaxed),
            s.seq.load(Ordering::Relaxed),
        )
    };
    let unpinned = filled().filter(|(_, s)| !pinned(s));
    let victim = match counts {
        None => unpinned.min_by_key(|(_, s)| tick(s)),
        Some(counts) => {
            // A transient (seq from 1) before every hot slot (seq 0).
            let cold = |s: &SlotBook| match s.seq.load(Ordering::Relaxed) {
                0 => u64::MAX,
                seq => seq,
            };
            let score = |s: &SlotBook| counts.score(s.id.load(Ordering::Relaxed));
            unpinned.min_by(|(_, a), (_, b)| {
                cold(a)
                    .cmp(&cold(b))
                    .then_with(|| score(a).total_cmp(&score(b)))
                    .then_with(|| tick(a).cmp(&tick(b)))
            })
        }
    };
    victim.map(|(k, _)| k).ok_or(if filled().next().is_some() {
        NoSlot::Pinned
    } else {
        NoSlot::Filling
    })
}

/// What one `ensure` claimed under the books' lock: the slots to fill, the
/// filled slots it let go (their pages to return) and the ids it did not
/// find filled.
struct Claimed {
    picks: Vec<Pick>,
    evicted: Vec<usize>,
    misses: u64,
    /// The ids found filled that the claim pinned as it found them
    /// ([`claim_slots_pinned`]), each one open read counted for the caller's
    /// [`BatchPin`] to close; none when the claim did not pin.
    pinned: Vec<u32>,
}

/// The books' pass of one `ensure` over layer `layer`'s `ids`: ids the plan's
/// host segment holds are skipped, a filled one is placed as `kind` says (a
/// fresh tick under [`Claim::Step`] and [`Claim::Keep`]; a hit under
/// [`Claim::Transient`] leaves the slot as it is, hot or transient), every other takes a slot — the first free, else the victim
/// ([`pick_slot`]), which it evicts — and is marked filling. No call's own
/// ids are its victims, including hits it has not visited yet. A transient
/// hit keeps its cold place and a hot hit its tick, so ids a keep claim took
/// stay newer. Under [`NvTierEvict::Heat`] a hit's count
/// moves only at the call's counting ([`Books::count_step`]), after the
/// picks. Pure over the
/// books: no read, no page returned. The `seen` marks are this call's alone;
/// a mark that outlived its call would skip an expert evicted since, and its
/// slot would read unfilled. Refused by name before any slot is claimed, the
/// books and the slots as they were: an id past the layer's experts, and a
/// call naming more distinct ids of the arena than the layer's slots
/// ([`TierError::TooManyIds`], the seam's own refusal) — a union reads every
/// id it names at once, so a claim past the slots would evict one of the
/// call's own ids before the call reads it. A miss that finds no slot to take
/// gives back what the call had taken and refuses: all slots filling is the
/// race refusal (two ensures raced the layer), all the rest pinned the
/// floor's.
fn claim_slots(
    layer: usize,
    arena: &LayerArena,
    books: &mut Books,
    ids: &[u32],
    kind: Claim,
) -> Result<Claimed, GpuError> {
    claim_with(layer, arena, books, ids, kind, false)
}

/// [`claim_slots`] that pins what it finds filled as it finds it, under the
/// caller's one hold of the books' lock: each hit counts one open read at
/// once, so no later miss of this call and no other claim takes it, and a
/// refused claim closes them again. The misses' slots are filling, which no
/// claim takes; the caller pins them as their fill publishes
/// ([`publish_pinned`]).
fn claim_slots_pinned(
    layer: usize,
    arena: &LayerArena,
    books: &mut Books,
    ids: &[u32],
    kind: Claim,
) -> Result<Claimed, GpuError> {
    claim_with(layer, arena, books, ids, kind, true)
}

fn claim_with(
    layer: usize,
    arena: &LayerArena,
    books: &mut Books,
    ids: &[u32],
    kind: Claim,
    pin: bool,
) -> Result<Claimed, GpuError> {
    let mut picks = Vec::new();
    let mut evicted = Vec::new();
    let mut pinned = Vec::new();
    let misses = claim_fill(
        layer,
        arena,
        books,
        ids,
        kind,
        pin,
        &mut picks,
        &mut evicted,
        &mut pinned,
    )?;
    Ok(Claimed {
        picks,
        evicted,
        misses,
        pinned,
    })
}

/// [`claim_with`] into the caller's buffers. A `pinned` buffer that already
/// has capacity is the pin's id buffer; a fresh empty one takes a buffer from
/// the arena's pool, as the allocating claim always did. On a refusal the
/// picks are given back and cleared, and the pin drops.
#[allow(
    clippy::too_many_arguments,
    reason = "the claim's pieces plus the three caller buffers the step slot reuses"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one claim: distinct ids, the slot take, the pin, and the refusal give-back"
)]
fn claim_fill(
    layer: usize,
    arena: &LayerArena,
    books: &mut Books,
    ids: &[u32],
    kind: Claim,
    pin: bool,
    picks: &mut Vec<Pick>,
    evicted: &mut Vec<usize>,
    pinned: &mut Vec<u32>,
) -> Result<u64, GpuError> {
    const WHAT: &str = "NvTier::ensure";
    let n = arena.n_expert as usize;
    for marks in [&mut books.seen, &mut books.named] {
        if marks.len() != n {
            marks.resize(n, false);
        }
        marks.fill(false);
    }
    let mut distinct = 0usize;
    for &id in ids {
        if id as usize >= n {
            return Err(GpuError::shape(
                WHAT,
                format!("expert {id} is past the {n} experts"),
            ));
        }
        if !arena.host.contains(&id) && !std::mem::replace(&mut books.named[id as usize], true) {
            distinct += 1;
        }
    }
    if distinct > arena.slots.len() {
        return Err(GpuError::from(model::ModelError::Tier(
            TierError::TooManyIds {
                layer,
                ids: distinct,
                slots: arena.slots.len(),
            },
        )));
    }
    picks.clear();
    evicted.clear();
    let pin_ids = if pin {
        if pinned.capacity() > 0 {
            let mut buf = std::mem::take(pinned);
            buf.clear();
            buf
        } else {
            arena.take_ids()
        }
    } else {
        Vec::new()
    };
    pinned.clear();
    let mut misses = 0u64;
    // Closes the hits' pins when the claim is refused.
    let mut held = BatchPin {
        arena,
        layer,
        ids: pin_ids,
    };
    for &id in ids {
        let at = id as usize;
        if arena.host.contains(&id) || std::mem::replace(&mut books.seen[at], true) {
            continue;
        }
        if let Some(slot) = arena.filled(id) {
            if pin {
                held.add(id, WHAT)?;
            }
            let book = &arena.slots[slot];
            if kind != Claim::Transient {
                books.tick += 1;
                book.tick.store(books.tick, Ordering::Relaxed);
                book.seq.store(0, Ordering::Relaxed);
            }
            continue;
        }
        misses += 1;
        let counts = books.counts.as_ref().map(|c| &c[layer]);
        let own = |s: &SlotBook| books.named[s.id.load(Ordering::Relaxed) as usize];
        let slot = match pick_slot(&arena.slots, |s| arena.pins(s) || own(s), counts) {
            Ok(slot) => slot,
            Err(why) => {
                unclaim(arena, picks);
                picks.clear();
                return Err(match why {
                    NoSlot::Filling => {
                        GpuError::shape(WHAT, "every slot is filling: two ensures raced a layer")
                    }
                    NoSlot::Pinned => GpuError::shape(
                        WHAT,
                        format!(
                            "layer {layer}: no slot is free and every filled one is pinned by a \
                             lane's read or a batch pin: the arena's {} slots do not cover the \
                             pins open beside a call",
                            arena.slots.len()
                        ),
                    ),
                });
            }
        };
        let book = &arena.slots[slot];
        let gone = (book.state.load(Ordering::Relaxed) == FILLED).then(|| {
            let gone = book.id.load(Ordering::Relaxed);
            arena.of_id[gone as usize].store(0, Ordering::Relaxed);
            evicted.push(slot);
            gone
        });
        book.state.store(FILLING, Ordering::Relaxed);
        book.id.store(id, Ordering::Relaxed);
        arena.of_id[at].store(slot as u32 + 1, Ordering::Relaxed);
        picks.push(Pick { slot, id, gone });
    }
    *pinned = std::mem::take(&mut held.ids);
    Ok(misses)
}

/// Give back what a refused claim took, last pick first: a slot that held an
/// id holds it filled again, with the tick and sequence word it had (a claim
/// writes none before its fill publishes), and a free one is free.
fn unclaim(arena: &LayerArena, picks: &[Pick]) {
    for p in picks.iter().rev() {
        let book = &arena.slots[p.slot];
        arena.of_id[p.id as usize].store(0, Ordering::Relaxed);
        match p.gone {
            Some(gone) => {
                book.id.store(gone, Ordering::Relaxed);
                book.state.store(FILLED, Ordering::Relaxed);
                arena.of_id[gone as usize].store(p.slot as u32 + 1, Ordering::Relaxed);
            }
            None => book.state.store(FREE, Ordering::Relaxed),
        }
    }
}

/// The picks a fill read, published under the books' lock: each slot's place
/// in the books as `kind` says — a fresh tick and no sequence word, or tick 0
/// and the next sequence number, the oldest transient the least — then
/// filled.
fn publish(arena: &LayerArena, books: &mut Books, picks: &[Pick], kind: Claim) {
    for p in picks {
        let book = &arena.slots[p.slot];
        match kind {
            Claim::Transient => {
                books.seq += 1;
                book.tick.store(0, Ordering::Relaxed);
                book.seq.store(books.seq, Ordering::Relaxed);
            }
            Claim::Step | Claim::Keep => {
                books.tick += 1;
                book.tick.store(books.tick, Ordering::Relaxed);
                book.seq.store(0, Ordering::Relaxed);
            }
        }
        book.state.store(FILLED, Ordering::Relaxed);
    }
}

/// [`publish`] and the pin of the picks, under the one hold of the books'
/// lock the caller has: each pick counts one open read as its slot turns
/// filled, so no claim sees it filled and unpinned. `held` is the batch's pin
/// with the hits [`claim_slots_pinned`] counted.
fn publish_pinned(
    arena: &LayerArena,
    books: &mut Books,
    picks: &[Pick],
    kind: Claim,
    held: &mut BatchPin<'_>,
) -> Result<(), GpuError> {
    publish(arena, books, picks, kind);
    for p in picks {
        held.add(p.id, "NvTier::ensure_pinned")?;
    }
    Ok(())
}

/// Where part `part` of the expert slot at byte `slot_off` of the mapping
/// holds `r`'s bytes: the logical bytes' offset in the mapping (the window's
/// base plus the codes' skew) and their length, the codes then the block
/// scales the fill moved after them.
fn window_span(
    slot_off: usize,
    windows: &[usize; PARTS],
    part: usize,
    r: PartRange,
) -> (usize, usize) {
    (slot_off + windows[part] + r.skew, r.len())
}

/// Each paged layer's byte offset in the mapping: the layers' `count` slots
/// laid one after the other, a layer's `slot_sizes` entry a slot. The
/// offsets never overlap and the end is `count × Σ slot_sizes`.
fn layer_bases(count: usize, slot_sizes: &[usize]) -> Vec<usize> {
    let mut at = 0;
    slot_sizes
        .iter()
        .map(|&b| {
            let base = at;
            at += count * b;
            base
        })
        .collect()
}

/// The offsets an audit reads of one part's span of `len` bytes at file
/// offset `at`: the first and the last page of it, the part's own first and
/// last bytes, and one page between — six sites a torn or zeroed slot
/// cannot all pass.
#[must_use]
pub fn audit_offsets(at: u64, len: usize) -> [usize; 6] {
    let page = DIRECT_ALIGN;
    let last = len.saturating_sub(1);
    let mid = (at as usize + len / 2) & !(page - 1);
    [
        0,
        page,
        len.saturating_sub(page),
        len / 2,
        mid.saturating_sub(at as usize),
        last,
    ]
    .map(|o| o.min(last))
}

/// Whether the arena's own address ranges (`[start, end)`, the spans its
/// evictions return) and the model mapping's are disjoint: an eviction
/// never advises the model mapping, so a lock's pinned pages and an r8
/// copy's pages stay theirs — the tier's drops of the mapping's pages are
/// [`NvTier::drop_layer`]'s and [`NvTier::end_read`]'s, inside
/// [`NvTier::drop_region`]. A mutant whose
/// eviction madvises a model page names an arena range that overlaps the
/// mapping's, and this turns false.
#[must_use]
pub fn advice_disjoint(arena: &[(usize, usize)], model: &[(usize, usize)]) -> bool {
    arena
        .iter()
        .all(|&(a0, a1)| model.iter().all(|&(m0, m1)| a1 <= m0 || m1 <= a0))
}

/// What a drop names over one paged layer ([`NvTier::drop_region`]): per
/// stack row of the plan, the split's shard and the whole-page byte runs
/// (file offsets) there.
pub type DropRuns = Vec<(usize, Vec<Range<u64>>)>;

/// `marks` with every id the books neither hold nor read marked read: a
/// drop of all of them.
fn read_all(marks: Vec<Mark>) -> Vec<Mark> {
    marks
        .into_iter()
        .map(|m| match m {
            Mark::Rest => Mark::Read,
            kept => kept,
        })
        .collect()
}

/// One layer of the arena: its slots and their books.
struct LayerArena {
    /// The plan's host segment for the layer: the ids the file mapping
    /// serves. Every other id is the arena's.
    host: Range<u32>,
    n_expert: u32,
    /// The plan's routed stack rows the layer pages, the first `rows` of
    /// each array below: three for a gated expert, two for a relu² one.
    rows: usize,
    /// Slot `k`'s part `p` window at `windows[p]` bytes into the slot.
    windows: [usize; PARTS],
    slot_bytes: usize,
    /// The layer's first byte in the mapping ([`layer_bases`]).
    base: usize,
    /// The mapping's first byte as the card addresses it, on a pinned arena.
    dev: Option<u64>,
    /// The plan's stack row each of the leg's parts reads, resolved at the
    /// attach by name ([`NvTier::note_parts`]); unread until then, and
    /// every pick of the layer refused by name after.
    part_of: OnceLock<[usize; PARTS]>,
    /// The fills' handle of each of the split's shards a run of the arena's
    /// lies in, by shard; a drop advises the shard's mapping.
    files: Arc<[Option<Arc<DirectFile>>]>,
    /// The plan's stack-row tensors' names, for the attach's mapping.
    names: [String; PARTS],
    /// Per id: its parts' runs, by stack row.
    parts: Vec<[PartRange; PARTS]>,
    /// Per id: the slot that holds it, + 1; 0 when none does.
    of_id: Vec<AtomicU32>,
    /// Per id: the lane reads of it open now ([`NvTier::open_read`]); a drop
    /// keeps its pages and the LRU its slot while it is nonzero.
    reading: Vec<AtomicU32>,
    /// Per id: of those, the reads that went to the model file's mapping,
    /// because its slot was not filled when they opened; the rest read the
    /// slot. Only the mapping's reads leave pages for a drop.
    mapped: Vec<AtomicU32>,
    /// Per slot: its state (a pick's read), its id and LRU tick (the
    /// lock's alone).
    slots: Vec<SlotBook>,
    /// The id buffers idle for the next pin ([`LayerArena::take_ids`]), each
    /// with capacity for the layer's slot count, the most ids one pin can
    /// hold; [`id_pool`] fills it at the carve, so a pin makes no allocator
    /// call. [`PIN_BUFFERS`] is the pins one layer can hold at once: a
    /// step's (`crates/model/src/moe.rs:2444`, `:2611`), two union batches
    /// (the computing batch's pin and the next batch's ticket, `:1768`,
    /// `:1780`; an unbatched union holds one, `:1694`), and a card hold a
    /// row of the step port's page ([`CardHold`]: a pass's rows each hold
    /// the last layer until their next go).
    /// A pin past those takes a fresh buffer that joins the pool when it
    /// drops; it is never refused.
    idle: Mutex<Vec<Vec<u32>>>,
}

/// The id buffers cut for one layer's pool at the carve ([`LayerArena::idle`]).
const PIN_BUFFERS: usize = 3 + super::page::MAX_ROWS;

/// [`LayerArena::idle`] for a layer of `slots` slots: [`PIN_BUFFERS`] buffers
/// of that capacity, the list itself sized so a buffer's return never grows
/// it.
fn id_pool(slots: usize) -> Mutex<Vec<Vec<u32>>> {
    let mut pool = Vec::with_capacity(PIN_BUFFERS);
    pool.extend((0..PIN_BUFFERS).map(|_| Vec::with_capacity(slots)));
    Mutex::new(pool)
}

impl LayerArena {
    /// Slot `slot`'s first byte in the mapping.
    fn slot_off(&self, slot: usize) -> usize {
        self.base + slot * self.slot_bytes
    }

    /// Where expert `id`'s bytes of the plan's stack row `row` lie in slot
    /// `slot`: their first byte in the mapping and their length. The one
    /// reading of a part's place in the mapping, which a pin's host bytes
    /// and card address ([`LayerArena::part_at`]), the lane's slot read
    /// ([`NvTier::staged_slot`]) and the audit share.
    fn span_of(&self, slot: usize, id: u32, row: usize) -> (usize, usize) {
        window_span(
            self.slot_off(slot),
            &self.windows,
            row,
            self.parts[id as usize][row],
        )
    }

    /// Where part `part` (as [`TierSlots::slot`] counts parts) of expert
    /// `id`'s filled slot lies in the mapping: its first byte and length.
    /// `None` before the attach resolved the parts, past them, and for an id
    /// with no filled slot. The host's bytes of a slot and the card's
    /// address of it read it alike.
    fn part_at(&self, id: u32, part: usize) -> Option<(usize, usize)> {
        let row = *self.part_of.get()?.get(part)?;
        let slot = self.filled(id)?;
        Some(self.span_of(slot, id, row))
    }

    /// Whether every part of an id the layer serves starts on a 4-byte word
    /// of the mapping in any slot: the layer's first byte, the slot size,
    /// each part's window and each such id's skew are multiples of four
    /// ([`LayerArena::span_of`]).
    fn on_words(&self) -> bool {
        let word = |n: usize| n.is_multiple_of(4);
        word(self.base)
            && word(self.slot_bytes)
            && self.windows.iter().all(|&w| word(w))
            && (0..self.n_expert)
                .filter(|id| !self.host.contains(id))
                .all(|id| self.parts_of(id).iter().all(|r| word(r.skew)))
    }

    /// Whether a lane's open read pins slot `s`: the expert it holds is read
    /// now. The LRU takes no pinned slot.
    fn pins(&self, s: &SlotBook) -> bool {
        self.reading
            .get(s.id.load(Ordering::Relaxed) as usize)
            .is_some_and(|n| n.load(Ordering::Acquire) > 0)
    }

    /// The slot holding expert `id` filled, if one does — the one reading
    /// of the books a pick, the residency seam and `ensure`'s hit check
    /// share. `id` past the layer's experts holds none.
    fn filled(&self, id: u32) -> Option<usize> {
        let held = self.of_id.get(id as usize)?.load(Ordering::Relaxed);
        let slot = (held as usize).checked_sub(1)?;
        (self.slots[slot].state.load(Ordering::Relaxed) == FILLED).then_some(slot)
    }

    /// Expert `id`'s slot as the books hold it: hot, or a transient (a
    /// sequence word), or `None` when `id` has no filled slot.
    fn heat(&self, id: u32) -> Option<Heat> {
        let book = &self.slots[self.filled(id)?];
        Some(if book.seq.load(Ordering::Relaxed) == 0 {
            Heat::Hot
        } else {
            Heat::Transient
        })
    }

    /// A drop's books with nothing read yet: the host segment's ids held,
    /// every other id the arena's.
    fn marks(&self) -> Vec<Mark> {
        (0..self.n_expert)
            .map(|id| {
                if self.host.contains(&id) {
                    Mark::Held
                } else {
                    Mark::Rest
                }
            })
            .collect()
    }

    /// [`LayerArena::marks`] with the ids a lane has open held too: no drop
    /// names a page of bytes a lane has yet to copy out.
    fn live_marks(&self) -> Vec<Mark> {
        let mut marks = self.marks();
        for (mark, open) in marks.iter_mut().zip(&self.reading) {
            if open.load(Ordering::Acquire) > 0 {
                *mark = Mark::Held;
            }
        }
        marks
    }

    /// The books of a whole layer's drop ([`NvTier::drop_layer`]): every id
    /// the arena serves read, but the ones a lane has open.
    fn layer_marks(&self) -> Vec<Mark> {
        read_all(self.live_marks())
    }

    /// Open a lane's read of expert `id` (layer `layer`, for the refusals):
    /// refused by name as `what` past the layer's experts.
    fn open(&self, layer: usize, id: u32, what: &'static str) -> Result<(), GpuError> {
        self.reading(layer, id, what)?
            .fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Whether the reads open of expert `id` are slot reads, not the
    /// mapping's: a read opened unfilled is counted in `mapped`, so any open
    /// read past those found the slot filled, and a batch pin counts as one.
    /// Refused by name as `what` past the layer's experts and with no read
    /// open.
    fn slot_read(&self, layer: usize, id: u32, what: &'static str) -> Result<bool, GpuError> {
        let open = self.reading(layer, id, what)?.load(Ordering::Acquire);
        if open == 0 {
            return Err(GpuError::protocol(
                what,
                format!("layer {layer} expert {id}: a slot read with no read open"),
            ));
        }
        Ok(open > self.mapped[id as usize].load(Ordering::Acquire))
    }

    /// Pin the filled slots of `ids` against the victim pick until the guard
    /// drops: each distinct id the arena serves counts one in its open reads,
    /// as a lane's open read of a filled slot does. All or none: refused by
    /// name as `what` for an id past the layer's experts and for one with no
    /// filled slot — a pin on an unfilled id would read as a lane's mapping
    /// read. The caller holds the books' lock, so no claim sees the ids half
    /// pinned or takes a slot between the check and the count.
    fn pin_batch<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        what: &'static str,
    ) -> Result<BatchPin<'a>, GpuError> {
        // Distinct and ascending, in a buffer of the layer's pool: a pin
        // holds no more ids than the layer has slots.
        let mut held = self.take_ids();
        for &id in ids.iter().filter(|id| !self.host.contains(id)) {
            if let Err(at) = held.binary_search(&id) {
                held.insert(at, id);
            }
        }
        let checked = held.iter().try_for_each(|&id| {
            self.reading(layer, id, what)?;
            if self.filled(id).is_none() {
                return Err(GpuError::protocol(
                    what,
                    format!("layer {layer} expert {id}: a batch pin of an unfilled slot"),
                ));
            }
            Ok(())
        });
        if let Err(e) = checked {
            self.give_ids(held);
            return Err(e);
        }
        for &id in &held {
            self.open(layer, id, what)?;
        }
        Ok(BatchPin {
            arena: self,
            layer,
            ids: held,
        })
    }

    /// A buffer for a pin's ids, empty, from the layer's pool; a fresh one
    /// when the pool is empty, which joins the pool when its pin drops.
    fn take_ids(&self) -> Vec<u32> {
        let pooled = self.idle.lock().unwrap_or_else(|e| e.into_inner()).pop();
        pooled.unwrap_or_else(|| Vec::with_capacity(self.slots.len()))
    }

    /// `ids` back into the pool, cleared. A buffer that never held capacity
    /// (a guard's moved-out stand-in) is not kept.
    fn give_ids(&self, mut ids: Vec<u32>) {
        ids.clear();
        if ids.capacity() > 0 {
            self.idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(ids);
        }
    }

    /// Close the reads a pin of `ids` opened and give the buffer back to the
    /// pool, whatever refused on the way. A count already at zero is a
    /// broken invariant (a read ended that was never opened), so this panics
    /// by name once it has closed the rest, unless the thread is already
    /// unwinding.
    fn unpin(&self, layer: usize, ids: Vec<u32>) {
        let mut broken = None;
        for &id in &ids {
            if let Err(e) = self.close(layer, id, "NvTier::pin") {
                broken.get_or_insert(e);
            }
        }
        self.give_ids(ids);
        if let Some(e) = broken
            && !std::thread::panicking()
        {
            panic!("{e}");
        }
    }

    /// Close a lane's read of expert `id` ([`LayerArena::open`]): refused by
    /// name as `what` past the layer's experts and with no read open — never
    /// a count wrapped past zero.
    fn close(&self, layer: usize, id: u32, what: &'static str) -> Result<(), GpuError> {
        self.reading(layer, id, what)?
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .map(|_| ())
            .map_err(|_| {
                GpuError::protocol(
                    what,
                    format!("layer {layer} expert {id}: a read ended that no lane opened"),
                )
            })
    }

    /// Expert `id`'s open-read count, refused by name as `what` past the
    /// layer's experts.
    fn reading(&self, layer: usize, id: u32, what: &'static str) -> Result<&AtomicU32, GpuError> {
        self.reading.get(id as usize).ok_or_else(|| {
            GpuError::shape(
                what,
                format!(
                    "layer {layer} expert {id} is past the {} experts",
                    self.n_expert
                ),
            )
        })
    }

    /// Expert `id`'s parts, by stack row.
    fn parts_of(&self, id: u32) -> &[PartRange] {
        &self.parts[id as usize][..self.rows]
    }

    /// The fills' handle of shard `shard`. Every run the build resolved has
    /// one ([`NvTier::of_paged`]).
    fn file(&self, shard: usize) -> &Arc<DirectFile> {
        self.files
            .get(shard)
            .and_then(Option::as_ref)
            .expect("a run's shard opened at the build")
    }

    /// The stack row and the expert whose run in shard `shard` holds the
    /// page of byte `at`, for a refusal to name; `None` when no run does.
    fn owner_at(&self, shard: usize, at: u64, page: u64) -> Option<(usize, u32)> {
        (0..self.n_expert).find_map(|id| {
            self.parts_of(id).iter().enumerate().find_map(|(row, p)| {
                p.runs()
                    .any(|r| {
                        r.shard == shard
                            && r.at / page * page <= at
                            && at < (r.at + r.len as u64).div_ceil(page) * page
                    })
                    .then_some((row, id))
            })
        })
    }
}

/// A batch of pinned ids ([`LayerArena::pin_batch`]): the drop closes the
/// reads it opened ([`LayerArena::unpin`]) and gives the id buffer back.
struct BatchPin<'a> {
    arena: &'a LayerArena,
    layer: usize,
    ids: Vec<u32>,
}

impl BatchPin<'_> {
    /// Count one open read of expert `id` into the batch, closed by the drop.
    fn add(&mut self, id: u32, what: &'static str) -> Result<(), GpuError> {
        self.arena.open(self.layer, id, what)?;
        self.ids.push(id);
        Ok(())
    }

    /// The pin as the seam holds it: the ids move into the token, whose
    /// release closes them ([`LayerArena`]'s [`PinRelease`]), and this
    /// guard's drop finds none to close.
    fn into_token(mut self) -> PinToken {
        PinToken {
            layer: self.layer,
            ids: std::mem::take(&mut self.ids),
        }
    }
}

impl Drop for BatchPin<'_> {
    fn drop(&mut self) {
        self.arena.unpin(self.layer, std::mem::take(&mut self.ids));
    }
}

impl PinRelease for LayerArena {
    fn release(&self, token: PinToken) {
        self.unpin(token.layer, token.ids);
    }

    fn device_slot(&self, id: u32, part: usize) -> Option<u64> {
        let dev = self.dev?;
        let (at, _) = self.part_at(id, part)?;
        Some(dev + at as u64)
    }
}

struct SlotBook {
    state: AtomicU32,
    /// The slot's expert, +1 semantics none: written under the books'
    /// lock, read by an evicting pass.
    id: AtomicU32,
    /// The slot's LRU tick: written under the books' lock, read by the
    /// victim pick. A transient's is 0, below every hot tick.
    tick: AtomicU64,
    /// A transient's place among the transients, oldest first, from 1; 0 for
    /// a hot slot. Written under the books' lock, read by the victim pick
    /// and the hit rule.
    seq: AtomicU64,
}

/// What the books picked for one miss: the slot to fill, the id the call
/// wants in it and, when the slot held an expert, the id it let go.
struct Pick {
    slot: usize,
    id: u32,
    gone: Option<u32>,
}

/// Ids of one layer the card leg reads, pinned in their slots
/// ([`NvTier::hold_for_card`], or [`CardTicket::finish`]): no claim takes their slots and
/// [`NvTier::release`] refuses them until the hold drops, which closes the
/// reads and gives the id buffer back. It owns its tier, so the step port
/// keeps it past the service that took it.
#[must_use = "the slots stay pinned until the hold drops"]
pub struct CardHold {
    tier: Arc<NvTier>,
    token: Option<PinToken>,
}

impl CardHold {
    /// The held ids, in the order the hold was asked for them.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        self.token.as_ref().map_or(&[], |t| &t.ids)
    }

    /// Where the card reads held expert `id` in the arena
    /// ([`NvTier::card_arena`]): the word offsets of its gate, up and down
    /// parts, as [`TierSlots::slot`] counts parts. Refused by name for an id
    /// the hold does not pin and for a part that does not start on a word.
    pub(crate) fn offsets(&self, id: u32) -> Result<[u64; PARTS], GpuError> {
        const WHAT: &str = "CardHold::offsets";
        let token = self
            .token
            .as_ref()
            .ok_or_else(|| GpuError::state(WHAT, "a hold that has not been dropped"))?;
        if !token.ids.contains(&id) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {} expert {id}: the hold pins {:?}",
                    token.layer, token.ids
                ),
            ));
        }
        self.tier.card_offsets(token.layer, id)
    }
}

impl Drop for CardHold {
    fn drop(&mut self) {
        if let Some(t) = self.token.take()
            && let Some(arena) = self.tier.by_layer.get(t.layer).and_then(|l| l.as_ref())
        {
            arena.unpin(t.layer, t.ids);
        }
    }
}

/// A card hold's claim in flight ([`NvTier::issue_card_hold`]): the ids'
/// slots are taken in the books and their misses are being filled on the
/// tier's reader. Nothing of the slots is readable through it: the card's
/// offsets and bytes are the [`CardHold`] that [`CardTicket::finish`] returns
/// once every slot is filled, so a reader of an unfilled slot has no hold to
/// read through. Dropped without a finish, the claim goes back when the fill
/// has ended.
#[must_use = "the claim lasts until the ticket is finished or dropped"]
pub(crate) struct CardTicket<'a> {
    tier: &'a Arc<NvTier>,
    layer: usize,
    ids: &'a [u32],
    ticket: Ticket<'a>,
}

impl CardTicket<'_> {
    /// Run `work`, which must read none of the claim's slots, and count its
    /// wall as host compute under a fill when the fill is still in flight at
    /// its start ([`NvTierStats::step_hits_under_read_ns`]). A ready ticket
    /// counts nothing.
    pub(crate) fn under_fill<R>(&self, work: impl FnOnce() -> R) -> R {
        if self.ticket.is_ready() {
            return work();
        }
        let t0 = Instant::now();
        let r = work();
        self.tier
            .stats
            .note_hits_under_read(t0.elapsed().as_nanos() as u64);
        r
    }

    /// Join the fill and pin the ids: the hold, naming its ids in the order
    /// the claim asked them (the order the step port writes their offsets
    /// in). A failed fill, or an id left unfilled, is refused by name.
    pub(crate) fn finish(self) -> Result<CardHold, GpuError> {
        const WHAT: &str = "NvTier::hold_for_card";
        let CardTicket {
            tier,
            layer,
            ids,
            ticket,
        } = self;
        let ready = ticket
            .wait()
            .map_err(|e| GpuError::from(model::ModelError::Tier(e)))?;
        let mut held = {
            let _books = tier.books.lock().unwrap_or_else(|e| e.into_inner());
            tier.arena_of(layer, WHAT)?.pin_batch(layer, ids, WHAT)?
        };
        drop(ready);
        if held.ids.len() != ids.len() || ids.iter().any(|id| !held.ids.contains(id)) {
            return Err(GpuError::state(WHAT, "a pin of exactly the asked ids"));
        }
        held.ids.copy_from_slice(ids);
        if tier.audit.get().is_some() {
            tier.audit_slots(layer, ids)?;
        }
        let token = held.into_token();
        tier.stats
            .card_ids
            .fetch_add(token.ids.len() as u64, Ordering::Relaxed);
        Ok(CardHold {
            tier: Arc::clone(tier),
            token: Some(token),
        })
    }
}

/// The tier's counters ([`NvTier::stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvTierStats {
    /// Ids `ensure` did not find filled.
    pub misses: u64,
    /// Slots filled.
    pub fills: u64,
    /// Bytes the fills read, the parts' own bytes.
    pub fill_bytes: u64,
    /// Wall the fills took, summed (ns) — beside the bytes, the fill rate.
    pub fill_ns: u64,
    /// Slots evicted for a later miss.
    pub evictions: u64,
    /// Bytes the arena holds filled — never above the budget.
    pub resident_bytes: u64,
    /// Reads served by the page cache ([`DirectFile::open_buffered`]).
    pub buffered_reads: u64,
    pub buffered_bytes: u64,
    /// Drops of the mapping's pages reads brought in
    /// ([`NvTier::drop_layer`], [`NvTier::end_read`]): the calls that named
    /// an id of the arena's.
    pub drops: u64,
    /// Bytes those drops named to the kernel, whole pages — the pages the
    /// reads faulted and the ones they did not alike.
    pub drop_bytes: u64,
    /// Wall those drops took, summed (ns): the syscalls and the kernel's
    /// freeing of the pages, on the reader's thread.
    pub drop_ns: u64,
    /// The slots each paged layer holds: at least the arena's floor.
    pub slots_per_layer: u64,
    /// Lane reads of an id the arena serves that opened it filled, pinned
    /// its slot and read it ([`NvTier::open_read`]); and the parts of them
    /// the arena lent ([`NvTier::staged_slot`]).
    pub slot_reads: u64,
    pub slot_parts: u64,
    /// Lane reads of an id the arena serves that opened it unfilled and read
    /// the model file's mapping, each queuing a drop at its end.
    pub mapping_reads: u64,
    /// Slots freed because the id went to the card ([`NvTier::release`]), and
    /// the releases of an id the arena held no filled slot of.
    pub releases: u64,
    pub release_misses: u64,
    /// Prompt union calls on a layer with an arena, by path: those whose
    /// arena ids fit the layer's slots less two batches (claimed whole) and
    /// those run in batches ([`TierSlots::note_union`]).
    pub union_small: u64,
    pub union_batched: u64,
    /// Ids the prompt union's pinned claims ([`NvTier::ensure_pinned`] under
    /// [`Claim::Keep`] and [`Claim::Transient`]) filled, and ids they found
    /// filled.
    pub union_fills: u64,
    pub union_hits: u64,
    /// Ids the pinned claims placed as [`Claim::Keep`] and as
    /// [`Claim::Transient`]; a pinned [`Claim::Step`] counts in neither.
    pub keep_claims: u64,
    pub transient_claims: u64,
    /// Ids the pinned [`Claim::Step`] claims (the step port's) filled, and
    /// ids they found filled: counted here, not in the union's.
    pub step_fills: u64,
    pub step_hits: u64,
    /// Step layers whose claim had at least one miss: one a layer, not a
    /// pick. A layer of hits, and a victim fill, are not this.
    pub step_miss_layers: u64,
    /// Wall of those step fills, as the step call measured it (ns): the
    /// pool's and the reader's, beside [`NvTierStats::fill_ns`].
    pub step_fill_wall_ns: u64,
    /// Host compute of a step's hit group, or of the card leg's host list
    /// ([`CardTicket::under_fill`]), while the miss ticket was still in
    /// flight (ns). A ready ticket and a fill wait are not this.
    pub step_hits_under_read_ns: u64,
    /// Ids the step port held for the card leg ([`NvTier::hold_for_card`]),
    /// which the card read from their slots.
    pub card_ids: u64,
}

/// The atomically counted [`NvTierStats`]: a counter a field.
#[derive(Default)]
struct Stats {
    misses: AtomicU64,
    fills: AtomicU64,
    fill_bytes: AtomicU64,
    fill_ns: AtomicU64,
    evictions: AtomicU64,
    resident_bytes: AtomicU64,
    buffered_reads: AtomicU64,
    buffered_bytes: AtomicU64,
    slot_reads: AtomicU64,
    slot_parts: AtomicU64,
    mapping_reads: AtomicU64,
    releases: AtomicU64,
    release_misses: AtomicU64,
    union_small: AtomicU64,
    union_batched: AtomicU64,
    union_fills: AtomicU64,
    union_hits: AtomicU64,
    keep_claims: AtomicU64,
    transient_claims: AtomicU64,
    step_fills: AtomicU64,
    step_hits: AtomicU64,
    step_miss_layers: AtomicU64,
    step_fill_wall_ns: AtomicU64,
    step_hits_under_read_ns: AtomicU64,
    card_ids: AtomicU64,
    /// The last union call of each layer, kept when asked for
    /// ([`NvTier::record_unions`]); unset while off.
    unions: OnceLock<Mutex<Vec<Option<UnionBook>>>>,
}

impl Stats {
    /// A pinned claim's ids as the books found them: `hits` filled already,
    /// `misses` to fill, placed as `kind` says. The hits go to the union's
    /// counter under [`Claim::Keep`] and [`Claim::Transient`], to the step
    /// port's under [`Claim::Step`].
    fn pinned_claim(&self, kind: Claim, hits: u64, misses: u64) {
        let (hit_counter, of_kind) = match kind {
            Claim::Keep => (&self.union_hits, Some(&self.keep_claims)),
            Claim::Transient => (&self.union_hits, Some(&self.transient_claims)),
            Claim::Step => (&self.step_hits, None),
        };
        hit_counter.fetch_add(hits, Ordering::Relaxed);
        if let Some(of_kind) = of_kind {
            of_kind.fetch_add(hits + misses, Ordering::Relaxed);
        }
    }

    /// A claim's misses, its pinned hits when the claim pinned, and the
    /// resident bytes an eviction gives back. The sync claim and the reused
    /// step job share it.
    fn note_claim(
        &self,
        arena: &LayerArena,
        kind: Claim,
        hits: Option<u64>,
        misses: u64,
        evictions: u64,
    ) {
        self.misses.fetch_add(misses, Ordering::Relaxed);
        if let Some(hits) = hits {
            self.pinned_claim(kind, hits, misses);
        }
        if evictions > 0 {
            self.evictions.fetch_add(evictions, Ordering::Relaxed);
            self.resident_bytes
                .fetch_sub(evictions * arena.slot_bytes as u64, Ordering::Relaxed);
        }
    }

    /// One union call on a layer with an arena, by the path it took.
    fn union_call(&self, batched: bool) {
        let by_path = if batched {
            &self.union_batched
        } else {
            &self.union_small
        };
        by_path.fetch_add(1, Ordering::Relaxed);
    }

    /// One step layer whose claim had a miss. A keep or transient claim,
    /// and a victim fill, are not this.
    fn note_step_miss(&self, kind: Claim) {
        if kind == Claim::Step {
            self.step_miss_layers.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The wall a step fill already measured. Not folded into `fill_ns`.
    fn note_step_wall(&self, kind: Claim, wall_ns: u64) {
        if kind == Claim::Step {
            self.step_fill_wall_ns.fetch_add(wall_ns, Ordering::Relaxed);
        }
    }

    /// Host compute of a step's hit group, or of the card leg's host list,
    /// while its miss fill was in flight.
    fn note_hits_under_read(&self, ns: u64) {
        self.step_hits_under_read_ns
            .fetch_add(ns, Ordering::Relaxed);
    }
}

/// A layer's last union call as the tier was told it
/// ([`TierSlots::note_union`], kept by [`NvTier::record_unions`]): the path it
/// took and, for every id of the call the arena serves, ascending, the
/// columns that listed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionBook {
    pub batched: bool,
    pub cols: Vec<(u32, u32)>,
}

impl UnionBook {
    /// The book of a call over `ids` (distinct, ascending) whose id `d` is
    /// listed by `off[d + 1] - off[d]` columns, less the ids of the plan's
    /// host segment `host`.
    fn of(host: &Range<u32>, ids: &[u32], off: &[usize], batched: bool) -> UnionBook {
        let cols = ids
            .iter()
            .zip(off.windows(2))
            .filter(|(id, _)| !host.contains(id))
            .map(|(&id, w)| (id, (w[1] - w[0]) as u32))
            .collect();
        UnionBook { batched, cols }
    }
}

/// A filled slot's place in the books ([`NvTier::slot_heat`]): hot, with a
/// tick of its own, or a transient at the cold end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heat {
    Hot,
    Transient,
}

/// What the tier's dropper thread shares with the tier: the split whose
/// shards' mappings it advises, the queue (one request a layer: the runs a
/// reader named, a later request replacing it), the first failure, and the
/// drops' counters.
struct DropShared {
    split: Arc<Split>,
    queue: Mutex<DropQueue>,
    /// Signalled when a request arrives or the tier stops.
    work: Condvar,
    /// Signalled when a request is done.
    done: Condvar,
    drops: AtomicU64,
    drop_bytes: AtomicU64,
    drop_ns: AtomicU64,
    #[cfg(test)]
    release_pause: Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>>,
}

struct DropQueue {
    /// Per layer: the runs a request named, until the dropper takes them.
    pending: Vec<Option<DropRuns>>,
    queued: usize,
    /// The first drop the kernel refused, named; every later request and
    /// flush returns it.
    failed: Option<String>,
    stop: bool,
}

impl DropShared {
    fn queue(&self) -> MutexGuard<'_, DropQueue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Out of the mapping and the page cache: `runs`, layer `layer`'s, and
    /// the drop counted.
    fn release(&self, layer: usize, runs: &DropRuns) -> Result<(), String> {
        #[cfg(test)]
        if let Some((entered, resume)) = self.release_pause.lock().unwrap().take() {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        let t0 = Instant::now();
        let mut pages = PageDrop::new(&self.split);
        for (shard, runs) in runs {
            for run in runs {
                pages
                    .release(*shard, run.clone())
                    .map_err(|e| format!("layer {layer} shard {shard}: {e}"))?;
            }
        }
        self.drops.fetch_add(1, Ordering::Relaxed);
        self.drop_bytes.fetch_add(pages.bytes(), Ordering::Relaxed);
        self.drop_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The dropper thread: take a queued layer's runs, release them, until
    /// the tier stops.
    fn run(&self) {
        let mut q = self.queue();
        loop {
            if q.stop {
                return;
            }
            let Some(layer) = q.pending.iter().position(Option::is_some) else {
                q = self.work.wait(q).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            let runs = q.pending[layer].take().expect("the position found it");
            q.queued -= 1;
            // Read entry holds this lock while it updates a queued range.
            let r = self.release(layer, &runs);
            if let Err(e) = r {
                q.failed.get_or_insert(e);
            }
            self.done.notify_all();
            drop(q);
            q = self.queue();
        }
    }
}

/// The reads of the one reused step fill. The job's closure clones one part
/// out and drops the lock before the read.
struct StepBuf {
    mem: Arc<ArenaMem>,
    reads: Mutex<Vec<PickRead>>,
}

/// One step fill, reused. A steady decode issues one prefetch at a time, so
/// one slot covers it and the issue allocates nothing after the first build.
/// A second prefetch that arrives while this one is open takes the allocating
/// path instead of sharing the slot.
struct StepSlot {
    active: AtomicBool,
    buf: Arc<StepBuf>,
    job: Mutex<Arc<Job>>,
    picks: Mutex<Vec<Pick>>,
    evicted: Mutex<Vec<usize>>,
    /// The pin a step ticket holds until it is waited for or dropped: the
    /// same layer and ids a [`PinToken`] carries.
    held: Mutex<Option<PinToken>>,
    kind: Mutex<Claim>,
    layer: AtomicUsize,
}

impl StepSlot {
    fn new(mem: Arc<ArenaMem>, slots: usize) -> StepSlot {
        let buf = Arc::new(StepBuf {
            mem,
            reads: Mutex::new(Vec::with_capacity(slots)),
        });
        let job_buf = Arc::clone(&buf);
        let job = Arc::new(Job::new(Kind::Step, 0, move |unit| {
            step_read(&job_buf, unit)
        }));
        StepSlot {
            active: AtomicBool::new(false),
            buf,
            job: Mutex::new(job),
            picks: Mutex::new(Vec::with_capacity(slots)),
            evicted: Mutex::new(Vec::with_capacity(slots)),
            held: Mutex::new(None),
            kind: Mutex::new(Claim::Step),
            layer: AtomicUsize::new(0),
        }
    }

    fn try_begin(&self) -> bool {
        self.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn release(&self) {
        self.active.store(false, Ordering::Release);
    }
}

/// The RAM arena of the NVMe expert tier ([`NvTier::of_paged`]).
pub struct NvTier {
    /// The arena's mapping, shared with the fills' reads.
    mem: Arc<ArenaMem>,
    /// One arena a layer, indexed by the plan's layer; `None` on a layer
    /// the plan pages nothing of.
    by_layer: Vec<Option<LayerArena>>,
    /// The LRU tick and the dedup pass's marks, under one lock taken per
    /// `ensure`.
    books: Mutex<Books>,
    /// The victim rule the build was asked for; the books hold their
    /// routing counts exactly under [`NvTierEvict::Heat`], so a call that
    /// only counts takes no lock under [`NvTierEvict::Tick`].
    evict: NvTierEvict,
    stats: Stats,
    /// The arena the plan reserved and the load took ([`NvTier::budget`] is
    /// the second).
    arena: ArenaGrant,
    /// The slots each paged layer holds.
    slots: usize,
    /// One slot of every paged layer, the carve's bytes.
    slot_set: u64,
    /// The routed-expert bytes the plan pages on the NVMe tier.
    paged: u64,
    /// One buffered handle per shard, once [`NvTier::audit_on`] turned the
    /// audit on.
    audit: OnceLock<Vec<(PathBuf, std::fs::File)>>,
    /// The threads a decode step's misses and a prompt batch's fill run on
    /// ([`NvTier::prefetch_pinned`]).
    reader: Reader,
    /// The one-column step's reused fill. See [`StepSlot`].
    step: StepSlot,
    /// The split whose shards' mappings the readers read and the drops
    /// advise.
    split: Arc<Split>,
    /// The host's page in bytes, the unit a drop's runs round to.
    page: u64,
    /// The dropper thread's shared state, and the thread, joined on drop.
    drops: Arc<DropShared>,
    dropper: Mutex<Option<JoinHandle<()>>>,
    /// Whether the step port's card leg reads the arena (`None`) or why
    /// not, as the load noted it ([`NvTier::note_card_leg`]); unset until
    /// then. Set once at load through the shared tier, so a cell.
    card_leg: OnceLock<Option<LegOff>>,
}

impl Drop for NvTier {
    fn drop(&mut self) {
        self.drops.queue().stop = true;
        self.drops.work.notify_all();
        let handle = self
            .dropper
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(h) = handle {
            // A dropper that panicked has nothing left to release: the
            // mapping goes with the split.
            let _ = h.join();
        }
        // A ticket alive here is a defect named, never a hang: the threads
        // are joined within the bound, and what a fill holds is its own.
        if let Err(e) = self.reader.stop(READER_STOP_BOUND)
            && !std::thread::panicking()
        {
            panic!("{e}");
        }
    }
}

#[derive(Default)]
struct Books {
    tick: u64,
    /// The last transient's sequence number.
    seq: u64,
    /// The dedup pass's seen marks, reused across calls.
    seen: Vec<bool>,
    /// The ids the current call names, protected from its victim picks
    /// and counted under [`NvTierEvict::Heat`].
    named: Vec<bool>,
    /// The NVMe tier's routing counts, the one owner of the heat a miss
    /// evicts by under [`NvTierEvict::Heat`]; `None` under
    /// [`NvTierEvict::Tick`], which counts nothing. By the plan's layer (a
    /// layer the arena does not cover holds no expert), per expert, the
    /// columns that routed it, each halved every [`HEAT_HALF_LIFE`] columns
    /// the layer has served since. A step's claim counts each id it names
    /// once and moves the layer's clock one column ([`Books::count_step`]); a
    /// prompt's union call counts each id by the columns that list it and
    /// moves the clock by its columns ([`TierSlots::note_union`]); a prompt
    /// batch's claim counts nothing, its call did. An evicted expert keeps its
    /// count. Written under the books' lock, sized at the build, so a count
    /// allocates nothing. The card's residency rule keeps its own counts
    /// (`runtime::SwapRule`): its rows are a stream's kept rows, blended
    /// across streams and decayed at its plans, and the arena's are every
    /// column the arena serves, a prompt's included.
    counts: Option<Vec<LayerCounts>>,
}

impl Books {
    /// Count a step's claim on layer `layer`, as the claim left its marks in
    /// `named` (each id the arena serves that the call names, once): advance
    /// the layer's clock by `columns`, add each marked id's selections at
    /// their positions in `column_off` (one without offsets), and clear its mark.
    /// A residency victim's fill moves the clock none, since the card served
    /// its columns and the arena never saw them. Nothing under [`NvTierEvict::Tick`].
    fn count_step(
        &mut self,
        layer: usize,
        ids: &[u32],
        columns: u64,
        column_off: Option<(&[usize], &[usize])>,
    ) {
        let Some(counts) = self.counts.as_mut() else {
            return;
        };
        let layer = &mut counts[layer];
        layer.advance(columns);
        for (i, &id) in ids.iter().enumerate() {
            if std::mem::replace(&mut self.named[id as usize], false) {
                match column_off {
                    Some((off, positions)) => {
                        layer.add_columns(id, columns, &positions[off[i]..off[i + 1]])
                    }
                    None => layer.add(id, 1.0),
                }
            }
        }
    }
}

/// The half-life of a routing count ([`LayerCounts`]), in the columns its
/// layer serves.
const HEAT_HALF_LIFE: f64 = 32.0;

/// The half-lives a layer's clock runs past its scores' base before they are
/// scaled down to it ([`LayerCounts::advance`]): a score stays below
/// `2^HEAT_RESCALE` times the columns since, far inside an `f64`.
const HEAT_RESCALE: f64 = 64.0;

/// One layer's counts, each held as a score: a column at `since` columns past
/// the base adds `2^(since / HEAT_HALF_LIFE)`, so every score of the layer is
/// its decayed count times one factor, and two experts rank by their scores
/// with no decay applied.
struct LayerCounts {
    /// The columns the layer has served since its scores' base.
    since: u64,
    scores: Vec<f64>,
}

impl LayerCounts {
    fn new(experts: usize) -> LayerCounts {
        LayerCounts {
            since: 0,
            scores: vec![0.0; experts],
        }
    }

    /// What one column at the clock adds to a score.
    fn weight(&self) -> f64 {
        (self.since as f64 / HEAT_HALF_LIFE).exp2()
    }

    /// The clock `columns` on; past [`HEAT_RESCALE`] half-lives the base
    /// moves up to it, every score scaled down by the weight it reached.
    fn advance(&mut self, columns: u64) {
        self.since += columns;
        if self.since as f64 >= HEAT_RESCALE * HEAT_HALF_LIFE {
            let down = self.weight().recip();
            self.scores.iter_mut().for_each(|s| *s *= down);
            self.since = 0;
        }
    }

    /// `columns` columns of expert `id` at the clock.
    fn add(&mut self, id: u32, columns: f64) {
        self.scores[id as usize] += columns * self.weight();
    }

    /// Routed columns at their positions within a call ending at the clock.
    fn add_columns(&mut self, id: u32, columns: u64, positions: &[usize]) {
        let weight: f64 = positions
            .iter()
            .map(|&col| ((col as f64 + 1.0 - columns as f64) / HEAT_HALF_LIFE).exp2())
            .sum();
        self.add(id, weight);
    }

    /// Expert `id`'s rank among the layer's: its score.
    fn score(&self, id: u32) -> f64 {
        self.scores[id as usize]
    }

    /// Expert `id`'s decayed count at the clock.
    fn count(&self, id: u32) -> f64 {
        self.score(id) / self.weight()
    }
}

impl NvTier {
    /// The arena of `plan`'s NVMe expert tier over `split`'s shards, in
    /// `mem`'s memory, read `O_DIRECT` (`direct`) or through the page cache:
    /// `None` on a plan that pages no routed expert or took no arena bytes (`arena.granted`
    /// 0 — the mapping path as it stands). The slots' budget is the arena
    /// the load was granted (`plan.host.nvme_arena_bytes` less what the
    /// host's reading cut, [`HostNeed::grant`](model::placement::workstation::HostNeed::grant);
    /// a build without a placed load passes [`ArenaGrant::whole`] of the
    /// plan's), the same count a paged layer. Each expert's part is read
    /// from its runs ([`stack_runs`]): a stack's one run of its file, its
    /// experts cut evenly, or the matrices a checkpoint folder locates, each
    /// in any shard. Refused by name: a layer whose host segment's ids are
    /// not one run, whose routed stacks are not two or three rows the split
    /// holds, a stack no file holds or whose experts do not cut its bytes
    /// evenly, a run in a shard that is not its file's mapping
    /// ([`gguf::Gguf::file_backed`]: a drop would zero a copy), and a budget
    /// whose carve falls under the arena's floor ([`ArenaFloor::of`] at the
    /// served column: a plan whose arena was not the split dial's). A miss
    /// once no slot is free takes its victim as `evict` says ([`pick_slot`]).
    pub fn of_paged(
        plan: &Plan<'_>,
        split: &Arc<Split>,
        direct: bool,
        arena: ArenaGrant,
        mem: ArenaKind<'_>,
        evict: NvTierEvict,
    ) -> Result<Option<NvTier>, GpuError> {
        const WHAT: &str = "NvTier::of_paged";
        if plan.host.nvme_expert_bytes == 0 || arena.granted == 0 {
            return Ok(None);
        }
        let model = plan.model;
        // Per layer: its routed stack rows' tensor names in row order, and
        // its host segment's id run.
        let mut rows_of: Vec<Vec<&ModelTensor>> = (0..model.layers).map(|_| Vec::new()).collect();
        let mut host_of: Vec<Option<Range<u32>>> = (0..model.layers).map(|_| None).collect();
        for r in &plan.rows {
            let t = &model.tensors[r.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let Some(l) = t.layer else { continue };
            if !r.segments.iter().any(|s| s.device == Device::Nvme) {
                continue;
            }
            rows_of[l].push(t);
            if let Some(s) = r.segments.iter().find(|s| s.device == Device::Host) {
                let ids = s.experts.as_ref().ok_or(GpuError::shape(
                    WHAT,
                    format!("{}: a host segment of a routed stack with no list", t.name),
                ))?;
                if let Some((first, rest)) = ids.ids().split_first() {
                    let (first, last) = (*first, rest.last().copied().unwrap_or(*first));
                    if ids.len() as u32 != last - first + 1 {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "{}: the host segment's {} ids are not one run ({}..={})",
                                t.name,
                                ids.len(),
                                first,
                                last
                            ),
                        ));
                    }
                    let run = first..last + 1;
                    match &host_of[l] {
                        None => host_of[l] = Some(run),
                        Some(h) if h == &run => {}
                        Some(h) => {
                            return Err(GpuError::shape(
                                WHAT,
                                format!(
                                    "{}: layer {l}'s stacks hold two host segments, {}..{} and \
                                     {}..{}",
                                    t.name, h.start, h.end, run.start, run.end
                                ),
                            ));
                        }
                    }
                }
            }
        }
        // The shards' direct handles, by shard, opened as a run names them.
        let mut files: Vec<Option<Arc<DirectFile>>> = Vec::new();
        let mut arenas: Vec<Option<LayerArena>> = (0..model.layers).map(|_| None).collect();
        let mut slot_sizes = Vec::new();
        for (l, rows) in rows_of.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            if !(2..=PARTS).contains(&rows.len()) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l} holds {} routed stack rows, not a relu² expert's two or a \
                         gated one's three",
                        rows.len()
                    ),
                ));
            }
            // The layer's own expert count: a family's layers may hold different counts.
            let n_expert = u32::try_from(model.layer_experts(l)).map_err(|_| {
                GpuError::shape(WHAT, format!("layer {l}: its expert count passes u32"))
            })?;
            let n = n_expert as usize;
            let mut parts = vec![[PartRange::default(); PARTS]; n];
            let mut lens = [(0usize, 0usize); PARTS];
            for (p, t) in rows.iter().enumerate() {
                let runs = stack_runs(split, t, n_expert)?;
                for &(codes, scales) in &runs {
                    for r in std::iter::once(codes).chain(scales) {
                        open_shard(split, &mut files, r.shard, direct, &t.name)?;
                    }
                }
                lens[p] = runs.first().map_or((0, 0), |&(codes, scales)| {
                    (codes.len, scales.map_or(0, |s| s.len))
                });
                for (id, (codes, scales)) in runs.into_iter().enumerate() {
                    let (_, _, need) = aligned_span(codes.at, codes.len as u64);
                    parts[id][p] = PartRange {
                        codes,
                        scales,
                        skew: need - codes.len,
                    };
                }
            }
            let mut windows = [0usize; PARTS];
            let mut slot_bytes = 0;
            for (window, &(codes, scales)) in windows.iter_mut().zip(&lens[..rows.len()]) {
                *window = slot_bytes;
                slot_bytes += window_bytes(codes, scales);
            }
            slot_sizes.push(slot_bytes);
            arenas[l] = Some(LayerArena {
                host: host_of[l].clone().unwrap_or(0..0),
                n_expert,
                rows: rows.len(),
                windows,
                slot_bytes,
                base: 0,
                dev: None,
                part_of: OnceLock::new(),
                files: Arc::new([]),
                names: std::array::from_fn(|p| {
                    rows.get(p).map(|t| t.name.clone()).unwrap_or_default()
                }),
                parts,
                of_id: (0..n).map(|_| AtomicU32::new(0)).collect(),
                reading: (0..n).map(|_| AtomicU32::new(0)).collect(),
                mapped: (0..n).map(|_| AtomicU32::new(0)).collect(),
                slots: Vec::new(),
                idle: Mutex::new(Vec::new()),
            });
        }
        let files: Arc<[Option<Arc<DirectFile>>]> = files.into();
        for arena in arenas.iter_mut().flatten() {
            arena.files = Arc::clone(&files);
        }
        let least = ArenaFloor::of(plan).map_err(|e| GpuError::plan(WHAT, e))?;
        let count =
            slots_of(arena.granted, &slot_sizes, &least).map_err(|e| GpuError::shape(WHAT, e))?;
        let mut tier = NvTier::carve(
            arenas,
            &slot_sizes,
            count,
            arena,
            plan.host.nvme_expert_bytes,
            split,
            mem,
        )?;
        if evict == NvTierEvict::Heat {
            tier.count_routes();
        }
        Ok(Some(tier))
    }

    /// Rank the hot slots by routing counts ([`NvTierEvict::Heat`]): zero
    /// counts over the covered layers. Load-time, before any claim.
    fn count_routes(&mut self) {
        let books = self.books.get_mut().unwrap_or_else(|e| e.into_inner());
        books.counts = Some(
            self.by_layer
                .iter()
                .map(|a| LayerCounts::new(a.as_ref().map_or(0, |a| a.n_expert as usize)))
                .collect(),
        );
        self.evict = NvTierEvict::Heat;
    }

    /// The tier over `arenas`, each paged layer's books with no slots yet,
    /// at `count` slots of `slot_sizes` a paged layer: the arena's memory as
    /// `mem` asks, each layer's base, card address and slot books, the fill
    /// reader and the dropper over `split`'s mappings. The carve of
    /// [`NvTier::of_paged`], once the plan's checks passed; `arena` and
    /// `paged` are the grant and the paged bytes the reports name.
    fn carve(
        mut arenas: Vec<Option<LayerArena>>,
        slot_sizes: &[usize],
        count: usize,
        arena: ArenaGrant,
        paged: u64,
        split: &Arc<Split>,
        mem: ArenaKind<'_>,
    ) -> Result<NvTier, GpuError> {
        const WHAT: &str = "NvTier::of_paged";
        let page = page_bytes().map_err(|e| GpuError::plan(WHAT, e))?;
        let bytes = count * slot_sizes.iter().sum::<usize>();
        let mem = match mem {
            ArenaKind::Anon => ArenaMem::new(
                AnonMap::new(bytes)
                    .map_err(|e| GpuError::shape(WHAT, format!("the arena's {bytes} B: {e}")))?,
            ),
            ArenaKind::Provided(source) => ArenaMem::provided(source.alloc(bytes)?, bytes),
        };
        let bases = layer_bases(count, slot_sizes);
        for (arena, base) in arenas.iter_mut().flatten().zip(bases) {
            arena.base = base;
            arena.dev = mem.device_start();
            arena.slots = (0..count)
                .map(|_| SlotBook {
                    state: AtomicU32::new(FREE),
                    id: AtomicU32::new(u32::MAX),
                    tick: AtomicU64::new(0),
                    seq: AtomicU64::new(0),
                })
                .collect();
            arena.idle = id_pool(count);
        }
        let reader = Reader::start(NVTIER_FILL_THREADS, "nvtier-fill")?;
        let mem = Arc::new(mem);
        let step = StepSlot::new(Arc::clone(&mem), count);
        let drops = Arc::new(DropShared {
            split: Arc::clone(split),
            queue: Mutex::new(DropQueue {
                pending: vec![None; arenas.len()],
                queued: 0,
                failed: None,
                stop: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            drops: AtomicU64::new(0),
            drop_bytes: AtomicU64::new(0),
            drop_ns: AtomicU64::new(0),
            #[cfg(test)]
            release_pause: Mutex::new(None),
        });
        let dropper = std::thread::Builder::new()
            .name("nvtier-drop".into())
            .spawn({
                let drops = Arc::clone(&drops);
                move || drops.run()
            })
            .map_err(|e| GpuError::shape(WHAT, format!("the dropper thread: {e}")))?;
        Ok(NvTier {
            mem,
            reader,
            step,
            by_layer: arenas,
            books: Mutex::new(Books::default()),
            evict: NvTierEvict::Tick,
            stats: Stats::default(),
            arena,
            slots: count,
            slot_set: slot_sizes.iter().sum::<usize>() as u64,
            paged,
            audit: OnceLock::new(),
            split: Arc::clone(split),
            page,
            drops,
            dropper: Mutex::new(Some(dropper)),
            card_leg: OnceLock::new(),
        })
    }

    /// Whether `split` is the open of the model file whose mapping the
    /// tier's drops advise: a reader of another open maps pages of its own,
    /// which a drop would leave cached.
    #[must_use]
    pub fn drops_reach(&self, split: &Arc<Split>) -> bool {
        Arc::ptr_eq(&self.split, split)
    }

    /// Whether the arena covers `layer` (a paged layer with slots).
    #[must_use]
    pub fn covers(&self, layer: usize) -> bool {
        self.by_layer.get(layer).is_some_and(|l| l.is_some())
    }

    /// The plan's host segment of `layer`, as the leg's tier handle holds
    /// it; `0..0` — every id the arena's — on a layer without one.
    pub(crate) fn host_ids(&self, layer: usize) -> Range<u32> {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .map_or(0..0, |l| l.host.clone())
    }

    /// Whether the arena, not the file mapping, serves layer `layer`'s
    /// expert `id`: the layer is a paged one and the plan's host segment
    /// does not hold the id — the NVMe segment's ids and the card's
    /// victims. The one predicate `ensure`, the residency seam and the
    /// leg's pick share; an id it names false reads the mapping as it
    /// always did.
    #[must_use]
    pub fn serves(&self, layer: usize, id: u32) -> bool {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .is_some_and(|l| !l.host.contains(&id))
    }

    /// Whether the arena's books hold layer `layer`'s expert `id` filled:
    /// the residency seam's answer for an id the arena serves — never the
    /// page cache's residency.
    #[must_use]
    pub fn slot_filled(&self, layer: usize, id: u32) -> bool {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .is_some_and(|l| l.filled(id).is_some())
    }

    /// Whether the books hold layer `layer`'s expert `id` filled as a hot
    /// slot or as a transient; `None` when unfilled.
    #[must_use]
    pub fn slot_heat(&self, layer: usize, id: u32) -> Option<Heat> {
        self.by_layer.get(layer)?.as_ref()?.heat(id)
    }

    /// Layer `layer`'s expert `id`'s routing count at the layer's clock
    /// ([`LayerCounts`]), filled or not; `None` under [`NvTierEvict::Tick`],
    /// which counts nothing, and for a layer or an id the arena does not
    /// cover.
    #[must_use]
    pub fn route_count(&self, layer: usize, id: u32) -> Option<f64> {
        let books = self.books.lock().unwrap_or_else(|e| e.into_inner());
        let counts = books.counts.as_ref()?.get(layer)?;
        ((id as usize) < counts.scores.len()).then(|| counts.count(id))
    }

    /// Keep each layer's last union call from now on
    /// ([`NvTier::last_union`]): an instrument a gate turns on, once; a run
    /// that does not ask keeps nothing and takes no lock.
    pub fn record_unions(&self) {
        self.stats.unions.get_or_init(Mutex::default);
    }

    /// Layer `layer`'s last union call, when [`NvTier::record_unions`] is on
    /// and the layer had one.
    #[must_use]
    pub fn last_union(&self, layer: usize) -> Option<UnionBook> {
        let books = self.stats.unions.get()?;
        let books = books.lock().unwrap_or_else(|e| e.into_inner());
        books.get(layer)?.clone()
    }

    /// The arena as the leg's [`TierSlots`] handle.
    pub(crate) fn slots(self: &Arc<Self>) -> Arc<dyn TierSlots> {
        self.clone()
    }

    /// Resolve the host leg's part indices — the names of the layer's three
    /// routed stacks as the leg built it, `[gate, up, down]`, a relu² leg's
    /// gate slot holding its up again (`model::moe::HostLayer::stacks`) — to
    /// the plan's stack rows the arena carved, refusing by name a stack the
    /// arena does not hold, a name twice but that gate slot's, and a row no
    /// part names. Load-time only, once; until it runs every pick of the
    /// layer refuses by name.
    pub(crate) fn note_parts(&self, layer: usize, leg: &[&str; PARTS]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::note_parts";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(GpuError::state(WHAT, "the layer's arena"));
        };
        let names = &arena.names[..arena.rows];
        let mut part_of = [usize::MAX; PARTS];
        for (p, name) in leg.iter().enumerate() {
            let Some(at) = names.iter().position(|n| n == name) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {layer}'s stack {name} is not one of the plan's routed rows {names:?}"
                    ),
                ));
            };
            let up_as_gate = p == 1 && leg[0] == leg[1];
            if part_of.contains(&at) && !up_as_gate {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {layer}'s stacks name the plan's row {name} twice"),
                ));
            }
            part_of[p] = at;
        }
        if let Some(row) = (0..arena.rows).find(|row| !part_of.contains(row)) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer}'s stacks {leg:?} name none of the plan's row {}",
                    names[row]
                ),
            ));
        }
        arena
            .part_of
            .set(part_of)
            .map_err(|_| GpuError::state(WHAT, "the layer's part mapping, set twice"))
    }

    /// The fill reader's counters since the load: the jobs and picks that ran
    /// off the caller, and the waits that found their fill unfinished.
    #[must_use]
    pub fn reader_stats(&self) -> ReaderStats {
        self.reader.stats()
    }

    /// The arena's counters since the load.
    #[must_use]
    pub fn stats(&self) -> NvTierStats {
        let s = &self.stats;
        NvTierStats {
            misses: s.misses.load(Ordering::Relaxed),
            fills: s.fills.load(Ordering::Relaxed),
            fill_bytes: s.fill_bytes.load(Ordering::Relaxed),
            fill_ns: s.fill_ns.load(Ordering::Relaxed),
            evictions: s.evictions.load(Ordering::Relaxed),
            resident_bytes: s.resident_bytes.load(Ordering::Relaxed),
            buffered_reads: s.buffered_reads.load(Ordering::Relaxed),
            buffered_bytes: s.buffered_bytes.load(Ordering::Relaxed),
            drops: self.drops.drops.load(Ordering::Relaxed),
            drop_bytes: self.drops.drop_bytes.load(Ordering::Relaxed),
            drop_ns: self.drops.drop_ns.load(Ordering::Relaxed),
            slots_per_layer: self.slots as u64,
            slot_reads: s.slot_reads.load(Ordering::Relaxed),
            slot_parts: s.slot_parts.load(Ordering::Relaxed),
            mapping_reads: s.mapping_reads.load(Ordering::Relaxed),
            releases: s.releases.load(Ordering::Relaxed),
            release_misses: s.release_misses.load(Ordering::Relaxed),
            union_small: s.union_small.load(Ordering::Relaxed),
            union_batched: s.union_batched.load(Ordering::Relaxed),
            union_fills: s.union_fills.load(Ordering::Relaxed),
            union_hits: s.union_hits.load(Ordering::Relaxed),
            keep_claims: s.keep_claims.load(Ordering::Relaxed),
            transient_claims: s.transient_claims.load(Ordering::Relaxed),
            step_fills: s.step_fills.load(Ordering::Relaxed),
            step_hits: s.step_hits.load(Ordering::Relaxed),
            step_miss_layers: s.step_miss_layers.load(Ordering::Relaxed),
            step_fill_wall_ns: s.step_fill_wall_ns.load(Ordering::Relaxed),
            step_hits_under_read_ns: s.step_hits_under_read_ns.load(Ordering::Relaxed),
            card_ids: s.card_ids.load(Ordering::Relaxed),
        }
    }

    /// The arena's budget in bytes: what the load was granted.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.arena.granted
    }

    /// The arena the plan reserved and the load took; they differ when the
    /// host's reading cut the arena at load.
    #[must_use]
    pub fn arena(&self) -> ArenaGrant {
        self.arena
    }

    /// One slot of every paged layer as the arena carved it: the bytes
    /// each slot a layer adds, the placement's slot set over the layers
    /// this plan pages.
    #[must_use]
    pub fn slot_set(&self) -> u64 {
        self.slot_set
    }

    /// The arena's mapping as an address range `[start, end)`: the only
    /// pages its slot releases return (none on a pinned arena).
    #[must_use]
    pub fn range(&self) -> (usize, usize) {
        let start = self.mem.start();
        (start, start + self.mem.len())
    }

    /// The routed-expert bytes the plan pages on the NVMe tier, which the
    /// arena serves beside the card's victims.
    #[must_use]
    pub fn paged_bytes(&self) -> u64 {
        self.paged
    }

    /// A lane is about to read layer `layer`'s expert `id`. An id the arena
    /// holds filled is pinned in its slot until [`NvTier::end_read`]: the LRU
    /// takes no pinned slot, so [`NvTier::staged_slot`]'s bytes stay put. Any
    /// other id the arena serves is read through the model file's mapping,
    /// and until the end no drop names a page of it — the bytes it faults in
    /// stay until the lane has copied them out. The check and the pin are
    /// made under the books' lock, so a claim either sees the pin or ran
    /// before it. The drop queue lock covers the entry and every page
    /// release; a queued range is revised before this read begins.
    /// Nothing for an id the arena does not serve. Refused by
    /// name: a layer the arena does not cover and an id past its experts.
    pub fn open_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::open_read";
        let arena = self.arena_of(layer, WHAT)?;
        if arena.host.contains(&id) {
            return Ok(());
        }
        let mut q = self.drops.queue();
        let _books = self.books.lock().unwrap_or_else(|e| e.into_inner());
        let runs = if q.pending[layer].is_some() {
            let mut marks = arena.layer_marks();
            let mark = marks.get_mut(id as usize).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("expert {id} is past the {} experts", arena.n_expert),
                )
            })?;
            *mark = Mark::Held;
            Some(self.runs_of(arena, &marks, WHAT)?)
        } else {
            None
        };
        arena.open(layer, id, WHAT)?;
        if let Some(runs) = runs {
            q.pending[layer] = Some(runs);
        }
        if arena.filled(id).is_some() {
            self.stats.slot_reads.fetch_add(1, Ordering::Relaxed);
        } else {
            arena.mapped[id as usize].fetch_add(1, Ordering::AcqRel);
            self.stats.mapping_reads.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Part `stack` (the plan's routed stack's tensor name) of layer
    /// `layer`'s expert `id`, read from its slot: `Some` when the lane's open
    /// read ([`NvTier::open_read`]) pinned the filled slot, `None` when it
    /// opened the id unfilled (the mapping's) or the arena holds no such
    /// stack. The bytes are the file's, at the skew the fill wrote them. A
    /// read with no open one, or of a slot the pin no longer finds filled, is
    /// refused by name: never a stale read.
    pub fn staged_slot(
        &self,
        layer: usize,
        id: u32,
        stack: &str,
    ) -> Result<Option<&[u8]>, GpuError> {
        const WHAT: &str = "NvTier::staged_slot";
        let arena = self.arena_of(layer, WHAT)?;
        if arena.host.contains(&id) {
            return Ok(None);
        }
        if !arena.slot_read(layer, id, WHAT)? {
            return Ok(None);
        }
        let slot = arena.filled(id).ok_or_else(|| {
            GpuError::protocol(
                WHAT,
                format!("layer {layer} expert {id}: its pinned slot reads unfilled"),
            )
        })?;
        let Some(row) = arena.names[..arena.rows].iter().position(|n| n == stack) else {
            return Ok(None);
        };
        let (at, len) = arena.span_of(slot, id, row);
        self.stats.slot_parts.fetch_add(1, Ordering::Relaxed);
        Ok(Some(self.mem.slice(at, len)))
    }

    /// The lane that opened layer `layer`'s expert `id`
    /// ([`NvTier::open_read`]) has copied it out. A slot read unpins the slot
    /// and queues nothing. A mapping read lets the layer's ids the arena
    /// serves leave the mapping and the page cache, the copied id and what
    /// the kernel read around it alike, but the ones a lane still has open,
    /// whose own end asks again ([`NvTier::drop_layer`]'s rule, queued for
    /// the dropper). Refused by name: an end with no open, a dropper that
    /// failed, besides [`NvTier::open_read`]'s refusals.
    pub fn end_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::end_read";
        let arena = self.arena_of(layer, WHAT)?;
        if arena.host.contains(&id) {
            return Ok(());
        }
        arena.close(layer, id, WHAT)?;
        let mapped = arena.mapped[id as usize]
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_ok();
        if mapped {
            self.queue_drop(layer, arena, WHAT)
        } else {
            Ok(())
        }
    }

    /// Layer `layer`'s expert `id` went to the card: its slot is freed — the
    /// state back to free, the slot's pages returned (`MADV_DONTNEED`, the
    /// arena's own range only; a pinned arena keeps them) — and the bytes
    /// the slot takes returned, 0 for an id the arena serves no filled slot
    /// of (an id the plan's host
    /// segment holds, one never filled, one the LRU took since its copy
    /// staged). Done under the books' lock, so no claim takes the slot
    /// between its freeing and its pages' return. Refused by name: a layer
    /// the arena does not cover, an id past its experts, and an id a lane
    /// has open or a batch pins — the landing follows its copy's staging.
    pub fn release(&self, layer: usize, id: u32) -> Result<u64, GpuError> {
        const WHAT: &str = "NvTier::release";
        let arena = self.arena_of(layer, WHAT)?;
        if id >= arena.n_expert {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} expert {id} is past the {} experts",
                    arena.n_expert
                ),
            ));
        }
        if arena.host.contains(&id) {
            return Ok(0);
        }
        let _books = self.books.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = arena.filled(id) else {
            self.stats.release_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(0);
        };
        if arena.reading(layer, id, WHAT)?.load(Ordering::Acquire) > 0 {
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {layer} expert {id}: freed while a lane or a batch pin has it open"),
            ));
        }
        arena.slots[slot].state.store(FREE, Ordering::Relaxed);
        arena.of_id[id as usize].store(0, Ordering::Relaxed);
        self.stats
            .resident_bytes
            .fetch_sub(arena.slot_bytes as u64, Ordering::Relaxed);
        self.stats.releases.fetch_add(1, Ordering::Relaxed);
        self.mem
            .give_back(arena.slot_off(slot), arena.slot_bytes)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(arena.slot_bytes as u64)
    }

    /// Pin layer `layer`'s `ids`, each of which the arena must hold filled,
    /// until the guard drops: no claim takes a pinned id's slot, and
    /// [`NvTier::release`] refuses it. Done under the books' lock, so a claim
    /// either ran before the pin or sees it. Refused by name, nothing pinned:
    /// a layer the arena does not cover, an id past its experts and an id
    /// with no filled slot. A pin counts as a lane's open read of a filled
    /// slot, so [`NvTier::staged_slot`] reads the same beside one. Ids of the
    /// plan's host segment are the mapping's and pin nothing.
    pub fn pin(&self, layer: usize, ids: &[u32]) -> Result<TierPin<'_>, GpuError> {
        const WHAT: &str = "NvTier::pin";
        let arena = self.arena_of(layer, WHAT)?;
        let _books = self.books.lock().unwrap_or_else(|e| e.into_inner());
        arena
            .pin_batch(layer, ids, WHAT)
            .map(|held| TierPin::new(arena, held.into_token()))
    }

    /// Layer `layer`'s drop, on this thread: every id the arena serves
    /// ([`NvTier::serves`]) leaves this process's page tables, then the page
    /// cache ([`PageDrop`]) — the ids a reader read, the ones the kernel read
    /// around them, and the ones nobody read, at no cost but the walk —
    /// except the ids a lane has open ([`NvTier::open_read`]), whose own end
    /// asks again. The pages are the whole ones [`drop_runs`] names: never
    /// one that holds a byte of the plan's host segment, of an id a lane has
    /// open, or of the tensors beside a stack. The readers queue the same
    /// drop for the dropper ([`NvTier::end_read`]).
    /// Refused by name: a layer the arena does not cover, and a drop the
    /// kernel refuses.
    pub fn drop_layer(&self, layer: usize) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::drop_layer";
        let arena = self.arena_of(layer, WHAT)?;
        let _q = self.drops.queue();
        let marks = arena.layer_marks();
        if !marks.contains(&Mark::Read) {
            return Ok(());
        }
        let runs = self.runs_of(arena, &marks, WHAT)?;
        self.drops
            .release(layer, &runs)
            .map_err(|e| GpuError::shape(WHAT, e))
    }

    /// Wait until every queued drop has run. Refused by name: a dropper
    /// that failed.
    pub fn flush(&self) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::flush";
        let mut q = self.drops.queue();
        while q.queued > 0 {
            q = self.drops.done.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        match &q.failed {
            Some(e) => Err(GpuError::protocol(WHAT, format!("the tier's dropper: {e}"))),
            None => Ok(()),
        }
    }

    /// Queue layer `layer`'s drop of every id the arena serves but the ones
    /// a lane has open, as the books stand now: the runs replace a request
    /// of the layer still queued. Refused by name as `what`: a dropper that
    /// failed, and books a drop cannot be planned from.
    fn queue_drop(
        &self,
        layer: usize,
        arena: &LayerArena,
        what: &'static str,
    ) -> Result<(), GpuError> {
        let mut q = self.drops.queue();
        let marks = arena.layer_marks();
        if !marks.contains(&Mark::Read) {
            return Ok(());
        }
        let runs = self.runs_of(arena, &marks, what)?;
        if let Some(e) = &q.failed {
            return Err(GpuError::protocol(what, format!("the tier's dropper: {e}")));
        }
        let slot = q
            .pending
            .get_mut(layer)
            .ok_or(GpuError::state(what, "the layer's queue slot"))?;
        if slot.replace(runs).is_none() {
            q.queued += 1;
        }
        drop(q);
        self.drops.work.notify_one();
        Ok(())
    }

    /// Layer `layer`'s arena, refused by name as `what` on a layer the arena
    /// does not cover.
    fn arena_of(&self, layer: usize, what: &'static str) -> Result<&LayerArena, GpuError> {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .ok_or(GpuError::state(what, "the layer's arena"))
    }

    /// The runs a drop of every id the arena serves in layer `layer` names
    /// — the most of the layer's stacks the tier ever drops: per stack row
    /// of the plan, its shard and the whole-page byte runs (file offsets)
    /// [`drop_runs`] writes. The one definition the drops, the attach's check
    /// ([`NvTier::refuse_held_pages`]) and a gate that reads the pages share.
    /// Refused by name: a layer the arena does not cover.
    pub fn drop_region(&self, layer: usize) -> Result<DropRuns, GpuError> {
        const WHAT: &str = "NvTier::drop_region";
        let arena = self.arena_of(layer, WHAT)?;
        self.runs_of(arena, &read_all(arena.marks()), WHAT)
    }

    /// Refuse by name a host set that holds a page some drop of the arena's
    /// ids names ([`NvTier::drop_region`]): a churn pool on a paged stack —
    /// or any run the host keeps there — whose pages the drops would take
    /// from under it. Load-time only, where a load's residency source
    /// takes the tier ([`super::swap_source::FileSwap::attach_tier`]).
    pub fn refuse_held_pages(&self, set: &HostSet) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::refuse_held_pages";
        let held = set.runs();
        for (layer, arena) in self.by_layer.iter().enumerate() {
            let Some(arena) = arena else { continue };
            for (shard, runs) in self.drop_region(layer)? {
                let Some((_, pages)) = held.iter().find(|(f, _)| *f == HostFile::Shard(shard))
                else {
                    continue;
                };
                if let Some(at) = overlap_of(&runs, pages, self.page) {
                    let owner = arena.owner_at(shard, at.start, self.page).map_or_else(
                        || format!("shard {shard}"),
                        |(row, id)| format!("{} (expert {id})", arena.names[row]),
                    );
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {layer}: the host set holds bytes {}..{} of {owner}, which the \
                             NVMe tier drops after each read: a churn pool on a paged stack would \
                             lose its pages under the host",
                            at.start, at.end,
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Per stack row of `arena`, per run of a part (its codes, then a planes
    /// matrix's block scales) and per shard those runs lie in, ascending: the
    /// shard and the runs [`drop_runs`] names under `marks` over the experts'
    /// runs it holds, in file order, every byte outside them kept as a
    /// stack's neighbours are. A refusal of the books names `what` and the
    /// row.
    fn runs_of(
        &self,
        arena: &LayerArena,
        marks: &[Mark],
        what: &'static str,
    ) -> Result<DropRuns, GpuError> {
        let mut out = Vec::with_capacity(PARTS);
        let mut runs_in: Vec<(usize, Range<u64>, Mark)> = Vec::with_capacity(marks.len());
        let (mut ranges, mut kept) = (Vec::new(), Vec::new());
        for row in 0..arena.rows {
            for scales in [false, true] {
                runs_in.clear();
                runs_in.extend(arena.parts.iter().zip(marks).filter_map(|(p, &mark)| {
                    let r = if scales {
                        p[row].scales
                    } else {
                        Some(p[row].codes)
                    }?;
                    Some((r.shard, r.at..r.at + r.len as u64, mark))
                }));
                runs_in.sort_unstable_by_key(|(shard, at, _)| (*shard, at.start));
                for shard_runs in runs_in.chunk_by(|a, b| a.0 == b.0) {
                    ranges.clear();
                    ranges.extend(shard_runs.iter().map(|(_, at, _)| at.clone()));
                    kept.clear();
                    kept.extend(shard_runs.iter().map(|&(_, _, mark)| mark));
                    let mut runs = Vec::new();
                    drop_runs(&ranges, &kept, self.page, &mut runs)
                        .map_err(|e| GpuError::shape(what, format!("{}: {e}", arena.names[row])))?;
                    out.push((shard_runs[0].0, runs));
                }
            }
        }
        Ok(out)
    }

    /// Fill the slots of `layer`'s `ids` the arena does not already hold:
    /// the misses' parts read with [`DirectFile::read_span_uninit`] into their
    /// slots' windows, split over the resident pool, over the victims'
    /// bytes in place. A read
    /// that fails names its file, offset and errno; a slot whose fill
    /// failed is freed, never lent half-filled. An id the plan's host
    /// segment holds is the mapping's and is skipped ([`NvTier::serves`]).
    pub fn ensure(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        self.ensure_as(layer, ids, Claim::Step)
    }

    /// [`NvTier::ensure`] with the call placed in the books as `claim` says:
    /// [`Claim::Step`] and [`Claim::Keep`] take a fresh tick, [`Claim::Transient`]
    /// enters its misses at the cold end, where they are the first victims.
    pub fn ensure_as(&self, layer: usize, ids: &[u32], claim: Claim) -> Result<(), GpuError> {
        self.fill_missing(layer, ids, claim, false)?;
        if self.audit.get().is_some() {
            self.audit_slots(layer, ids)?;
        }
        Ok(())
    }

    /// The residency lane's victim fill: `ids` claimed as a step's (unpinned)
    /// and their misses read on the tier's fill reader at the victim rank,
    /// behind every queued prompt batch, then published unpinned. It waits on
    /// the reader and takes nothing of the pool, whose one dispatch mutex is
    /// the chain's: beside the chain the lane takes only the books' lock, for
    /// the claim and the publish, which wait on nothing. A victim holds no
    /// pin once published, and the fill counts as a fill, not a union's.
    /// Under [`NvTierEvict::Heat`] each victim counts one column and the
    /// layer's clock stays ([`Books::count_step`]). Refused by name like
    /// [`NvTier::ensure`], and once the reader has stopped.
    pub fn ensure_victim(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        let (arena, claim) = self.claim(layer, ids, Claim::Step, false, 0, None)?;
        if !claim.picks.is_empty() {
            let reads: Vec<PickRead> = claim.picks.iter().map(|p| read_of(arena, p)).collect();
            let mem = Arc::clone(&self.mem);
            let tier = TierParts {
                arena,
                books: &self.books,
                stats: &self.stats,
                reader: &self.reader,
            };
            fill_victims(tier, &claim.picks, move |pick, part| {
                read_part(&mem, &reads[pick], part)
            })?;
        }
        if self.audit.get().is_some() {
            self.audit_slots(layer, ids)?;
        }
        Ok(())
    }

    /// [`NvTier::ensure_as`] and [`NvTier::pin`] as one claim: every id of
    /// `ids` the arena serves is filled and pinned until the guard drops. A
    /// hit is pinned as the claim finds it and a miss as its fill publishes,
    /// each under one hold of the books' lock with the claim's or the
    /// publish's own, so no claim takes a slot of the batch between its
    /// claim and its pin. The misses are read on the resident pool, as
    /// [`NvTier::ensure_as`]'s. Refused by name like the two it joins, nothing
    /// pinned: a layer the arena does not cover, an id past its experts, a
    /// call past the slots and a miss that finds every slot taken.
    pub fn ensure_pinned(
        &self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
    ) -> Result<TierPin<'_>, GpuError> {
        let held = self.fill_missing(layer, ids, claim, true)?;
        if self.audit.get().is_some() {
            self.audit_slots(layer, ids)?;
        }
        Ok(TierPin::new(held.arena, held.into_token()))
    }

    /// [`NvTier::ensure_pinned`] under [`Claim::Step`] for the card leg: every
    /// id of `ids` the arena serves filled and pinned until the returned hold
    /// drops, which the step port keeps until the card has read the slots
    /// ([`super::step`]). `ids` are ids the arena serves, at most the layer's
    /// slots. An invalid claim is refused by name. The fill is joined here;
    /// [`NvTier::issue_card_hold`] leaves it in flight.
    pub fn hold_for_card(
        self: &Arc<Self>,
        layer: usize,
        ids: &[u32],
    ) -> Result<CardHold, GpuError> {
        self.issue_card_hold(layer, ids)?.finish()
    }

    /// [`NvTier::hold_for_card`] up to the fill: the slots of `ids` are
    /// claimed and their misses' reads queued on the tier's reader, and the
    /// caller has them back while they run. The caller computes what needs
    /// none of those bytes, then [`CardTicket::finish`] joins the fill and
    /// pins; the card's slot offsets and the card's reads are the hold's,
    /// never the ticket's. A ticket dropped without a finish gives its claim
    /// back once its fill has ended. `ids` are as for
    /// [`NvTier::hold_for_card`]; an invalid claim is refused by name.
    pub(crate) fn issue_card_hold<'a>(
        self: &'a Arc<Self>,
        layer: usize,
        ids: &'a [u32],
    ) -> Result<CardTicket<'a>, GpuError> {
        let ticket = TierSlots::prefetch_pinned(self.as_ref(), layer, ids, Claim::Step, 1, None)
            .map_err(|e| GpuError::from(model::ModelError::Tier(e)))?;
        Ok(CardTicket {
            tier: self,
            layer,
            ids,
            ticket,
        })
    }

    /// The arena as the card addresses it: its first byte's device address
    /// and its length in 4-byte words. `None` on an anonymous arena.
    #[must_use]
    pub fn card_arena(&self) -> Option<(u64, usize)> {
        Some((self.mem.device_start()?, self.mem.len() / 4))
    }

    /// Where the card leg reads layer `layer`'s expert `id` in the arena
    /// ([`NvTier::card_arena`]): the word offsets of its gate, up and down
    /// parts, as [`TierSlots::slot`] counts parts. Refused by name for an id
    /// with no filled slot and for a part that does not start on a word.
    /// Reached through [`CardHold::offsets`] only: a hold exists once its
    /// fill has ended.
    fn card_offsets(&self, layer: usize, id: u32) -> Result<[u64; PARTS], GpuError> {
        const WHAT: &str = "NvTier::card_offsets";
        let arena = self.arena_of(layer, WHAT)?;
        let mut offs = [0u64; PARTS];
        for (p, off) in offs.iter_mut().enumerate() {
            let (at, _) = arena.part_at(id, p).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("layer {layer} expert {id} part {p}: no filled slot holds it"),
                )
            })?;
            if !at.is_multiple_of(4) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {layer} expert {id} part {p} starts at byte {at}, off a word"),
                ));
            }
            *off = (at / 4) as u64;
        }
        Ok(offs)
    }

    /// Record whether the step port's card leg reads this arena: `None`, or
    /// why it is off. Load-time only, once: a second note is refused by
    /// name. [`NvTier::card_leg`] reads it.
    pub(crate) fn note_card_leg(&self, off: Option<LegOff>) -> Result<(), GpuError> {
        self.card_leg.set(off).map_err(|_| {
            GpuError::state("NvTier::note_card_leg", "the card leg's state, noted twice")
        })
    }

    /// `None` when the step port's card leg reads this arena, else why not;
    /// [`LegOff::Unattached`] when the load never asked for one.
    #[must_use]
    pub fn card_leg(&self) -> Option<LegOff> {
        *self.card_leg.get().unwrap_or(&Some(LegOff::Unattached))
    }

    /// Whether every part of an id layer `layer`'s arena serves starts on a
    /// 4-byte word in whichever slot holds it, as the card leg's arena
    /// entries read it ([`CardHold::offsets`]); false for a layer the
    /// arena does not cover. Load-time.
    #[must_use]
    pub fn parts_on_words(&self, layer: usize) -> bool {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .is_some_and(LayerArena::on_words)
    }

    /// Layer `layer`'s slot count; 0 for a layer the arena does not cover.
    #[must_use]
    pub fn layer_slots(&self, layer: usize) -> usize {
        self.by_layer
            .get(layer)
            .and_then(|l| l.as_ref())
            .map_or(0, |arena| arena.slots.len())
    }

    /// Turn the audit on, once: after every [`NvTier::ensure`] each slot it
    /// answers for is compared with a fresh buffered read of the same file
    /// range at the six sites of [`audit_offsets`] — a torn or zeroed span is
    /// refused by name just before the compute reads it. Opens one buffered
    /// handle per shard the arena reads; costs the sampled bytes per pick.
    pub fn audit_on(&self) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::audit_on";
        let mut files: Vec<(PathBuf, std::fs::File)> = Vec::new();
        for arena in self.by_layer.iter().flatten() {
            for f in arena.files.iter().flatten() {
                if files.iter().all(|(p, _)| p != f.path()) {
                    let open = std::fs::File::open(f.path()).map_err(|e| {
                        GpuError::shape(WHAT, format!("open {}: {e}", f.path().display()))
                    })?;
                    files.push((f.path().to_path_buf(), open));
                }
            }
        }
        self.audit
            .set(files)
            .map_err(|_| GpuError::state(WHAT, "an arena whose audit is off"))
    }

    /// The audit of `layer`'s `ids` ([`NvTier::audit_on`]): every id the
    /// arena serves, each part, each site.
    fn audit_slots(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::audit";
        let (Some(files), Some(arena)) = (
            self.audit.get(),
            self.by_layer.get(layer).and_then(|l| l.as_ref()),
        ) else {
            return Err(GpuError::state(WHAT, "the layer's arena with the audit on"));
        };
        let mut fresh = vec![0u8; DIRECT_ALIGN];
        for &id in ids.iter().filter(|&&id| !arena.host.contains(&id)) {
            let slot = arena.filled(id).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("layer {layer} expert {id}: unfilled after ensure"),
                )
            })?;
            for part in 0..arena.rows {
                let (at, len) = arena.span_of(slot, id, part);
                let mut lent = self.mem.slice(at, len);
                for r in arena.parts[id as usize][part].runs() {
                    let (held, rest) = lent.split_at(r.len);
                    lent = rest;
                    let path = arena.file(r.shard).path();
                    let (_, file) = files.iter().find(|(p, _)| p == path).ok_or_else(|| {
                        GpuError::shape(
                            WHAT,
                            format!("{} was not opened for the audit", path.display()),
                        )
                    })?;
                    for off in audit_offsets(r.at, r.len) {
                        let n = DIRECT_ALIGN.min(r.len - off);
                        let from = r.at + off as u64;
                        file.read_exact_at(&mut fresh[..n], from).map_err(|e| {
                            GpuError::shape(WHAT, format!("read {} at {from}: {e}", path.display()))
                        })?;
                        if fresh[..n] != held[off..off + n] {
                            return Err(GpuError::shape(
                                WHAT,
                                format!(
                                    "layer {layer} expert {id} part {part}: the slot's {n} B at \
                                     +{off} of its run are not the file's at {from}",
                                ),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Layer `layer`'s arena and the books' claim of `ids` as `kind`, the
    /// misses counted, and each evicted slot counted out of the resident
    /// bytes — the pass the sync fill and the reader's share. With `pin` the
    /// ids found filled are pinned as the claim finds them. A step's claim
    /// is counted with the clock `columns` on ([`Books::count_step`]).
    /// Refused by name: a layer the arena does not cover and the claim's own
    /// refusals.
    fn claim(
        &self,
        layer: usize,
        ids: &[u32],
        kind: Claim,
        pin: bool,
        columns: u64,
        column_off: Option<(&[usize], &[usize])>,
    ) -> Result<(&LayerArena, Claimed), GpuError> {
        const WHAT: &str = "NvTier::ensure";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(GpuError::state(WHAT, "the layer's arena"));
        };
        let claim = {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            let claim = if pin {
                claim_slots_pinned(layer, arena, &mut books, ids, kind)?
            } else {
                claim_slots(layer, arena, &mut books, ids, kind)?
            };
            if kind == Claim::Step {
                books.count_step(layer, ids, columns, column_off);
            }
            claim
        };
        self.stats.note_claim(
            arena,
            kind,
            pin.then_some(claim.pinned.len() as u64),
            claim.misses,
            claim.evicted.len() as u64,
        );
        Ok((arena, claim))
    }

    /// The misses of `ids` read into their slots ([`NvTier::ensure`]), the
    /// reads on the resident pool. With `pin` every id the arena serves is
    /// pinned too ([`NvTier::ensure_pinned`]) and the guard returned holds
    /// them; without, the guard holds none.
    fn fill_missing(
        &self,
        layer: usize,
        ids: &[u32],
        kind: Claim,
        pin: bool,
    ) -> Result<BatchPin<'_>, GpuError> {
        let (arena, claim) = self.claim(layer, ids, kind, pin, 1, None)?;
        // The hits' pins close if anything below refuses.
        let mut held = BatchPin {
            arena,
            layer,
            ids: claim.pinned,
        };
        let picks = claim.picks;
        if picks.is_empty() {
            return Ok(held);
        }
        // A step layer with a miss, pinned or not. The unpinned ensure's
        // fill is counted with no claim, so the wall is noted beside it.
        self.stats.note_step_miss(kind);
        let t0 = Instant::now();
        self.fill(arena, &picks)
            .inspect_err(|_| free_picks(arena, &picks))?;
        {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            if pin {
                publish_pinned(arena, &mut books, &picks, kind, &mut held)?;
            } else {
                publish(arena, &mut books, &picks, kind);
            }
        }
        let wall = t0.elapsed().as_nanos() as u64;
        count_fill(&self.stats, arena, &picks, wall, pin.then_some(kind));
        self.stats.note_step_wall(kind, wall);
        Ok(held)
    }

    /// The misses' reads on the resident pool, each worker reading the parts
    /// its chunk names, each into its own window of its slot. The three parts
    /// of a pick are disjoint, and the picks are disjoint by the books' slot
    /// ownership, so the parts run together. Joined before this returns; a
    /// pick is published only after that join, never on one or two parts.
    fn fill(&self, arena: &LayerArena, picks: &[Pick]) -> Result<(), GpuError> {
        let reads: Vec<PickRead> = picks.iter().map(|p| read_of(arena, p)).collect();
        let fail: Mutex<Option<GpuError>> = Mutex::new(None);
        let n = reads.len() * PARTS;
        threads::pool().for_each_chunk(n, |chunk| {
            for unit in chunk {
                let pick = unit / PARTS;
                let part = unit % PARTS;
                if let Err(e) = read_part(&self.mem, &reads[pick], part) {
                    return note(&fail, e);
                }
            }
        });
        fail.into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .map_or(Ok(()), Err)
    }

    /// The step slot's claim. `Ok(Some(pin))` is a batch with nothing to fill.
    /// `Ok(None)` has queued the reused job; [`NvTier::step_finish`] joins it.
    /// A refusal gives the claim back and leaves the slot's buffers empty.
    fn issue_step(
        &self,
        layer: usize,
        ids: &[u32],
        kind: Claim,
        columns: u64,
        column_off: Option<(&[usize], &[usize])>,
    ) -> Result<Option<TierPin<'_>>, TierError> {
        const WHAT: &str = "NvTier::ensure";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(seam_error(
                layer,
                ids,
                GpuError::state(WHAT, "the layer's arena"),
            ));
        };
        let mut picks = self.step.picks.lock().unwrap_or_else(|e| e.into_inner());
        let mut evicted = self.step.evicted.lock().unwrap_or_else(|e| e.into_inner());
        let mut pinned = Vec::new();
        let filled = {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            let filled = claim_fill(
                layer,
                arena,
                &mut books,
                ids,
                kind,
                true,
                &mut picks,
                &mut evicted,
                &mut pinned,
            );
            if filled.is_ok() && kind == Claim::Step {
                books.count_step(layer, ids, columns, column_off);
            }
            filled
        };
        let misses = match filled {
            Ok(misses) => misses,
            Err(e) => {
                picks.clear();
                evicted.clear();
                return Err(seam_error(layer, ids, e));
            }
        };
        self.stats.note_claim(
            arena,
            kind,
            Some(pinned.len() as u64),
            misses,
            evicted.len() as u64,
        );
        evicted.clear();
        drop(evicted);
        if picks.is_empty() {
            drop(picks);
            return Ok(Some(TierPin::new(arena, PinToken { layer, ids: pinned })));
        }
        self.stats.note_step_miss(kind);
        {
            let mut reads = self
                .step
                .buf
                .reads
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            reads.clear();
            for p in picks.iter() {
                reads.push(read_of(arena, p));
            }
        }
        let units = picks.len() * PARTS;
        drop(picks);
        *self.step.held.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(PinToken { layer, ids: pinned });
        *self.step.kind.lock().unwrap_or_else(|e| e.into_inner()) = kind;
        self.step.layer.store(layer, Ordering::Relaxed);
        let job = Arc::clone(&self.step.job.lock().unwrap_or_else(|e| e.into_inner()));
        job.rearm(Kind::of_claim(kind), units);
        if let Err(e) = self.reader.resubmit(&job) {
            let mut picks = self.step.picks.lock().unwrap_or_else(|e| e.into_inner());
            free_picks(arena, &picks);
            picks.clear();
            if let Some(held) = self
                .step
                .held
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                drop(BatchPin {
                    arena,
                    layer: held.layer,
                    ids: held.ids,
                });
            }
            return Err(seam_error(layer, ids, e));
        }
        Ok(None)
    }

    /// Join the step slot's fill, publish it, and return the pin. On a
    /// failure the picks are freed and the pin dropped.
    fn step_finish(&self) -> Result<TierPin<'_>, TierError> {
        let layer = self.step.layer.load(Ordering::Relaxed);
        let kind = *self.step.kind.lock().unwrap_or_else(|e| e.into_inner());
        let job = Arc::clone(&self.step.job.lock().unwrap_or_else(|e| e.into_inner()));
        let held = self
            .step
            .held
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("a step fill holds its pin");
        let arena = self
            .by_layer
            .get(held.layer)
            .and_then(|l| l.as_ref())
            .expect("a step fill's layer has an arena");
        let held = BatchPin {
            arena,
            layer: held.layer,
            ids: held.ids,
        };
        let mut picks = self.step.picks.lock().unwrap_or_else(|e| e.into_inner());
        let filled = settle_fill(
            TierParts {
                arena,
                books: &self.books,
                stats: &self.stats,
                reader: &self.reader,
            },
            kind,
            &picks,
            held,
            Some(job),
        );
        let result = (|| {
            let (held, wall) = filled.map_err(|f| fill_error(layer, f))?;
            if self.audit.get().is_some() {
                self.audit_slots(layer, &held.ids)
                    .map_err(|e| seam_error(layer, &held.ids, e))?;
            }
            if !picks.is_empty() {
                count_fill(&self.stats, arena, &picks, wall, Some(kind));
                self.stats.note_step_wall(kind, wall);
            }
            Ok(TierPin::new(arena, held.into_token()))
        })();
        picks.clear();
        result
    }

    /// [`NvTier::ensure_pinned`] as a claim in flight: the batch is claimed
    /// and its hits pinned on the caller, its misses' reads queued on the
    /// tier's reader at the rank its claim sets, and the [`Inflight`] returned
    /// holds the claim until [`Inflight::finish`] joins the fill and publishes
    /// it. A batch with no miss queues nothing and allocates no job: its
    /// pins are the batch's. Refused by name like [`NvTier::ensure_pinned`],
    /// and once the reader has stopped; nothing is pinned then.
    fn prefetch(
        &self,
        layer: usize,
        ids: &[u32],
        kind: Claim,
        columns: u64,
        column_off: Option<(&[usize], &[usize])>,
    ) -> Result<Prefetch<'_>, GpuError> {
        let (arena, claim) = self.claim(layer, ids, kind, true, columns, column_off)?;
        let held = BatchPin {
            arena,
            layer,
            ids: claim.pinned,
        };
        if claim.picks.is_empty() {
            return Ok(Prefetch::Held(held));
        }
        self.stats.note_step_miss(kind);
        let reads: Vec<PickRead> = claim.picks.iter().map(|p| read_of(arena, p)).collect();
        let mem = Arc::clone(&self.mem);
        Inflight::queue(
            TierParts {
                arena,
                books: &self.books,
                stats: &self.stats,
                reader: &self.reader,
            },
            held,
            claim.picks,
            kind,
            move |pick, part| read_part(&mem, &reads[pick], part),
        )
        .map(Prefetch::Filling)
    }
}

/// What [`NvTier::prefetch`] made of a claim: the pins of a batch with
/// nothing to fill, or the claim in flight.
enum Prefetch<'a> {
    Held(BatchPin<'a>),
    Filling(Inflight<'a>),
}

/// One pick's reads as a reader thread needs them, all owned: each part's
/// files, windows and aligned spans ([`read_of`]); `None` past the layer's
/// rows.
struct PickRead {
    id: u32,
    parts: [Option<PartRead>; PARTS],
}

/// One part's reads: its codes' run (a part's one run) into the part's
/// window and, for a matrix kept as two planes, its block scales' run into
/// the window after it ([`window_bytes`]), which the fill then moves to
/// follow the codes. Cloning clones the files' `Arc`s, not the files.
#[derive(Clone)]
struct PartRead {
    codes: RunRead,
    scales: Option<RunRead>,
}

/// One run's read: the file's aligned span from `start` into `span` bytes
/// of the mapping at `at`, of which the run needs the first `need`, its own
/// `len` bytes the last of those.
#[derive(Clone)]
struct RunRead {
    file: Arc<DirectFile>,
    at: usize,
    span: usize,
    start: u64,
    need: usize,
    len: usize,
}

impl RunRead {
    /// Run `r` of `file` read into the window at byte `at` of the mapping.
    fn of(file: &Arc<DirectFile>, at: usize, r: Run) -> RunRead {
        let (start, span, need) = aligned_span(r.at, r.len as u64);
        RunRead {
            file: Arc::clone(file),
            at,
            span,
            start,
            need,
            len: r.len,
        }
    }

    /// The run's first byte in the mapping once it is read.
    fn first(&self) -> usize {
        self.at + self.need - self.len
    }
}

/// Pick `p`'s reads in `arena`: each part into its own window of the pick's
/// slot, at the skew an aligned read of the part's file range needs; a
/// planes matrix's block scales into the window after its codes'.
fn read_of(arena: &LayerArena, p: &Pick) -> PickRead {
    let slot_off = arena.slot_off(p.slot);
    let parts = arena.parts_of(p.id);
    PickRead {
        id: p.id,
        parts: std::array::from_fn(|part| {
            let r = parts.get(part)?;
            let at = slot_off + arena.windows[part];
            let codes = RunRead::of(arena.file(r.codes.shard), at, r.codes);
            let scales = r
                .scales
                .map(|s| RunRead::of(arena.file(s.shard), at + window_bytes(r.codes.len, 0), s));
            Some(PartRead { codes, scales })
        }),
    }
}

/// One part of a pick, read into that part's windows; nothing for a part
/// the layer does not hold. The pick owns its slot (the books marked it
/// Filling, and no claim takes a filling slot) and this writes only this
/// part's windows of it. The parts are disjoint, so they run together.
fn read_part(mem: &ArenaMem, r: &PickRead, part: usize) -> Result<(), GpuError> {
    r.parts[part]
        .as_ref()
        .map_or(Ok(()), |p| read_part_of(mem, r.id, p))
}

/// Part `part` of expert `id`: its codes' run, then a planes matrix's block
/// scales and their move to follow the codes — the bytes the slot lends,
/// whole once this returns.
fn read_part_of(mem: &ArenaMem, id: u32, part: &PartRead) -> Result<(), GpuError> {
    read_run(mem, id, &part.codes)?;
    if let Some(scales) = &part.scales {
        read_run(mem, id, scales)?;
        mem.move_within(
            scales.first(),
            part.codes.first() + part.codes.len,
            scales.len,
        );
    }
    Ok(())
}

fn read_run(mem: &ArenaMem, id: u32, run: &RunRead) -> Result<(), GpuError> {
    const WHAT: &str = "NvTier::fill";
    let got = mem
        .read_into(&run.file, run.start, run.at, run.span)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    if got < run.need {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "expert {id}: the run ends {got} B in, its part needs {} B at {}",
                run.need, run.start
            ),
        ));
    }
    Ok(())
}

/// One unit of the reused step job. The pick's part is cloned out of the
/// buffer before the read, so the three parts of a fill do not share the lock.
fn step_read(buf: &StepBuf, unit: usize) -> Result<(), GpuError> {
    let (id, part) = {
        let reads = buf.reads.lock().unwrap_or_else(|e| e.into_inner());
        let Some(read) = reads.get(unit / PARTS) else {
            return Err(GpuError::protocol(
                "NvTier::fill",
                format!("unit {unit} is past the step fill's {} picks", reads.len()),
            ));
        };
        (read.id, read.parts[unit % PARTS].clone())
    };
    part.map_or(Ok(()), |part| read_part_of(&buf.mem, id, &part))
}

/// One pick's parts, read one after another on this thread. The pool
/// and the reader call [`read_part`] instead, one unit a part. A slot an
/// eviction handed over keeps its pages: the reads overwrite the bytes they
/// place and no page is returned and faulted in again, and the bytes of a
/// window beside the read stay the evicted expert's, which no reader touches
/// (a part's `need` is every byte the slot lends). The refill test is the
/// only caller.
#[cfg(test)]
fn read_pick(mem: &ArenaMem, r: &PickRead) -> Result<(), GpuError> {
    for part in 0..PARTS {
        read_part(mem, r, part)?;
    }
    Ok(())
}

/// Count a fill of `picks` that ended in `wall_ns`: the slots and bytes read,
/// the wall, the bytes the arena holds now, and the parts read through the
/// page cache. A pinned claim's fill (`claim` its kind) is also the union's
/// under [`Claim::Keep`] and [`Claim::Transient`], the step port's under
/// [`Claim::Step`]; `None`, an unpinned fill, is neither.
fn count_fill(
    stats: &Stats,
    arena: &LayerArena,
    picks: &[Pick],
    wall_ns: u64,
    claim: Option<Claim>,
) {
    let mut bytes = 0u64;
    for p in picks {
        for r in arena.parts_of(p.id).iter().flat_map(PartRange::runs) {
            bytes += r.len as u64;
            if !arena.file(r.shard).is_direct() {
                stats.buffered_reads.fetch_add(1, Ordering::Relaxed);
                stats
                    .buffered_bytes
                    .fetch_add(r.len as u64, Ordering::Relaxed);
            }
        }
    }
    stats.fills.fetch_add(picks.len() as u64, Ordering::Relaxed);
    let of_claim = match claim {
        Some(Claim::Keep | Claim::Transient) => Some(&stats.union_fills),
        Some(Claim::Step) => Some(&stats.step_fills),
        None => None,
    };
    if let Some(of_claim) = of_claim {
        of_claim.fetch_add(picks.len() as u64, Ordering::Relaxed);
    }
    stats.fill_bytes.fetch_add(bytes, Ordering::Relaxed);
    stats.fill_ns.fetch_add(wall_ns, Ordering::Relaxed);
    stats.resident_bytes.fetch_add(
        picks.len() as u64 * arena.slot_bytes as u64,
        Ordering::Relaxed,
    );
}

/// A failed fill frees its slots: never lent half-filled.
fn free_picks(arena: &LayerArena, picks: &[Pick]) {
    for p in picks {
        arena.slots[p.slot].state.store(FREE, Ordering::Relaxed);
        arena.of_id[p.id as usize].store(0, Ordering::Relaxed);
    }
}

/// What an [`Inflight`] borrows of its tier: the layer's arena, the books,
/// the counters and the reader.
#[derive(Clone, Copy)]
struct TierParts<'a> {
    arena: &'a LayerArena,
    books: &'a Mutex<Books>,
    stats: &'a Stats,
    reader: &'a Reader,
}

/// A batch's fill that failed: the id of the pick whose reads failed, and
/// what failed them.
struct FillFailed {
    id: u32,
    error: GpuError,
}

/// Wait for `job`, then publish `picks` under the books lock. A failed wait
/// or a failed publish frees the picks; the pin drops with this call's
/// error. The sync claim and the reused step job share it.
fn settle_fill<'a>(
    tier: TierParts<'a>,
    kind: Claim,
    picks: &[Pick],
    mut held: BatchPin<'a>,
    job: Option<Arc<Job>>,
) -> Result<(BatchPin<'a>, u64), FillFailed> {
    let first = picks.first().map_or(u32::MAX, |p| p.id);
    let wall = match job {
        Some(job) => tier.reader.wait(&job).map_err(|f| {
            free_picks(tier.arena, picks);
            FillFailed {
                id: picks[f.unit / PARTS].id,
                error: f.error,
            }
        })?,
        None => 0,
    };
    {
        let mut books = tier.books.lock().unwrap_or_else(|e| e.into_inner());
        publish_pinned(tier.arena, &mut books, picks, kind, &mut held).map_err(|error| {
            free_picks(tier.arena, picks);
            FillFailed { id: first, error }
        })?;
    }
    Ok((held, wall))
}

/// A batch's claim whose misses are read on the reader ([`NvTier::prefetch`]),
/// at the rank its claim sets ([`Kind::of_claim`]). It holds the hits' pins
/// and the picks' slots, filling; [`Inflight::finish`] joins the fill and publishes the picks under
/// the books' lock, each pinned as it turns filled, so no claim between the
/// claim and the pin takes a slot of the batch. A fill that failed frees its
/// slots and closes the pins. Dropped unfinished — a caller that refused for
/// another reason — it blocks until the fill ended, then gives the claim back:
/// the picks published (the bytes are the file's) and every pin closed. Its
/// reads own their data, so none outlives what they read.
struct Inflight<'a> {
    tier: TierParts<'a>,
    kind: Claim,
    picks: Vec<Pick>,
    /// The pins: the hits' from the claim, the picks' once published. None
    /// once settled.
    held: Option<BatchPin<'a>>,
    /// The queued fill; none when the batch had no miss.
    job: Option<Arc<Job>>,
}

impl<'a> Inflight<'a> {
    /// `picks`' reads queued as one job on the reader, `read(pick, part)` one
    /// part. The job is three units a pick, and the batch publishes only
    /// after every part has landed. A reader that has stopped refuses by
    /// name, the claim given back whole (the picks freed, the pins closed).
    fn queue(
        tier: TierParts<'a>,
        held: BatchPin<'a>,
        picks: Vec<Pick>,
        kind: Claim,
        read: impl Fn(usize, usize) -> Result<(), GpuError> + Send + Sync + 'static,
    ) -> Result<Inflight<'a>, GpuError> {
        let units = picks.len() * PARTS;
        let mut inflight = Inflight {
            tier,
            kind,
            picks,
            held: Some(held),
            job: None,
        };
        if units > 0 {
            match inflight
                .tier
                .reader
                .submit(Job::new(Kind::of_claim(kind), units, move |unit| {
                    read(unit / PARTS, unit % PARTS)
                })) {
                Ok(job) => inflight.job = Some(job),
                Err(e) => {
                    free_picks(inflight.tier.arena, &inflight.picks);
                    inflight.picks.clear();
                    inflight.held = None;
                    return Err(e);
                }
            }
        }
        Ok(inflight)
    }

    /// Join the fill and publish its picks, pinned: the batch's pin, or the
    /// fill's failure.
    fn finish(mut self) -> Result<BatchPin<'a>, FillFailed> {
        self.settle()
    }

    fn settle(&mut self) -> Result<BatchPin<'a>, FillFailed> {
        let held = self.held.take().expect("a claim settles once");
        let (held, wall) = settle_fill(self.tier, self.kind, &self.picks, held, self.job.take())?;
        if !self.picks.is_empty() {
            count_fill(
                self.tier.stats,
                self.tier.arena,
                &self.picks,
                wall,
                Some(self.kind),
            );
            self.tier.stats.note_step_wall(self.kind, wall);
        }
        Ok(held)
    }
}

impl Drop for Inflight<'_> {
    fn drop(&mut self) {
        if self.held.is_some() {
            let _ = self.settle();
        }
    }
}

/// A victim's picks, claimed unpinned, read on the reader as one job at the
/// victim rank (`read(pick, part)` one part), waited for and published
/// unpinned: nothing is held once this returns, and the fill is counted as a
/// fill, not a union's. A fill that failed, or a reader that has stopped,
/// frees the picks' slots and refuses by name. The job is three units a pick.
fn fill_victims(
    tier: TierParts<'_>,
    picks: &[Pick],
    read: impl Fn(usize, usize) -> Result<(), GpuError> + Send + Sync + 'static,
) -> Result<(), GpuError> {
    let TierParts {
        arena,
        books,
        stats,
        reader,
    } = tier;
    let t0 = Instant::now();
    let units = picks.len() * PARTS;
    let job = reader
        .submit(Job::new(Kind::Victim, units, move |unit| {
            read(unit / PARTS, unit % PARTS)
        }))
        .inspect_err(|_| free_picks(arena, picks))?;
    reader.wait(&job).map_err(|f| {
        free_picks(arena, picks);
        f.error
    })?;
    {
        let mut books = books.lock().unwrap_or_else(|e| e.into_inner());
        publish(arena, &mut books, picks, Claim::Step);
    }
    count_fill(stats, arena, picks, t0.elapsed().as_nanos() as u64, None);
    Ok(())
}

/// A fill that failed, as the seam names it to the model side.
fn fill_error(layer: usize, f: FillFailed) -> TierError {
    TierError::Fill {
        layer,
        id: f.id,
        source: Box::new(f.error),
    }
}

/// Record the first fill failure, the rest dropped on the floor.
fn note(fail: &Mutex<Option<GpuError>>, e: GpuError) {
    let mut cell = fail.lock().unwrap_or_else(|e| e.into_inner());
    cell.get_or_insert(e);
}

/// An `ensure` of layer `layer`'s `ids` refused with `e`, as the seam names
/// it to the model side: the books' refusal of a call past the layer's slots
/// passed on whole ([`TierError::TooManyIds`]), any other a fill's failure
/// naming the call's first id.
fn seam_error(layer: usize, ids: &[u32], e: GpuError) -> TierError {
    let fill = |e: GpuError| TierError::Fill {
        layer,
        id: ids.first().copied().unwrap_or(u32::MAX),
        source: Box::new(e),
    };
    match e {
        GpuError::Model(m) => match *m {
            model::ModelError::Tier(t @ TierError::TooManyIds { .. }) => t,
            other => fill(GpuError::Model(Box::new(other))),
        },
        e => fill(e),
    }
}

impl TierSlots for NvTier {
    fn ensure_as(&self, layer: usize, ids: &[u32], claim: Claim) -> Result<(), TierError> {
        NvTier::ensure_as(self, layer, ids, claim).map_err(|e| seam_error(layer, ids, e))
    }

    fn pin<'a>(&'a self, layer: usize, ids: &[u32]) -> Result<TierPin<'a>, TierError> {
        NvTier::pin(self, layer, ids).map_err(|e| TierError::Pin {
            layer,
            source: Box::new(e),
        })
    }

    fn slots(&self, layer: usize) -> usize {
        self.layer_slots(layer)
    }

    fn ensure_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
    ) -> Result<TierPin<'a>, TierError> {
        NvTier::ensure_pinned(self, layer, ids, claim).map_err(|e| seam_error(layer, ids, e))
    }

    fn prefetch_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
        columns: u64,
        column_off: Option<(&[usize], &[usize])>,
    ) -> Result<Ticket<'a>, TierError> {
        // The steady step reuses one slot and boxes nothing. A prefetch that
        // arrives while that slot is open keeps the allocating path.
        if self.step.try_begin() {
            match self.issue_step(layer, ids, claim, columns, column_off) {
                Ok(Some(pin)) => {
                    if self.audit.get().is_some() {
                        let audit_ids = pin.ids().to_vec();
                        if let Err(e) = self.audit_slots(layer, &audit_ids) {
                            drop(pin);
                            self.step.release();
                            return Err(seam_error(layer, &audit_ids, e));
                        }
                    }
                    self.step.release();
                    return Ok(Ticket::ready(pin));
                }
                Ok(None) => return Ok(Ticket::bound(self)),
                Err(e) => {
                    self.step.release();
                    return Err(e);
                }
            }
        }
        let inflight = match self
            .prefetch(layer, ids, claim, columns, column_off)
            .map_err(|e| seam_error(layer, ids, e))?
        {
            Prefetch::Held(held) => {
                if self.audit.get().is_some() {
                    self.audit_slots(layer, ids)
                        .map_err(|e| seam_error(layer, ids, e))?;
                }
                return Ok(Ticket::ready(TierPin::new(held.arena, held.into_token())));
            }
            Prefetch::Filling(inflight) => inflight,
        };
        let ids = ids.to_vec();
        Ok(Ticket::new(move || {
            let held = inflight.finish().map_err(|f| fill_error(layer, f))?;
            if self.audit.get().is_some() {
                self.audit_slots(layer, &ids)
                    .map_err(|e| seam_error(layer, &ids, e))?;
            }
            Ok(TierPin::new(held.arena, held.into_token()))
        }))
    }

    fn slot(&self, layer: usize, id: u32, part: usize) -> Option<&[u8]> {
        let (at, len) = self.by_layer.get(layer)?.as_ref()?.part_at(id, part)?;
        Some(self.mem.slice(at, len))
    }

    fn note_step_hits_under_read(&self, ns: u64) {
        self.stats.note_hits_under_read(ns);
    }

    fn note_union(
        &self,
        layer: usize,
        ids: &[u32],
        off: &[usize],
        positions: &[usize],
        cols: usize,
        batched: bool,
    ) {
        self.stats.union_call(batched);
        if self.evict == NvTierEvict::Heat
            && let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref())
        {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(counts) = books.counts.as_mut() {
                let layer = &mut counts[layer];
                layer.advance(cols as u64);
                for (&id, w) in ids.iter().zip(off.windows(2)) {
                    if !arena.host.contains(&id) {
                        layer.add_columns(id, cols as u64, &positions[w[0]..w[1]]);
                    }
                }
            }
        }
        if let Some(books) = self.stats.unions.get() {
            let book = UnionBook::of(&self.host_ids(layer), ids, off, batched);
            let mut books = books.lock().unwrap_or_else(|e| e.into_inner());
            if books.len() <= layer {
                books.resize(layer + 1, None);
            }
            books[layer] = Some(book);
        }
    }
}

impl<'a> TicketWait<'a> for NvTier {
    fn finish_ticket(&'a self) -> Result<TierPin<'a>, TierError> {
        let pin = self.step_finish();
        self.step.release();
        pin
    }

    fn abandon_ticket(&'a self) {
        if let Ok(pin) = self.finish_ticket() {
            drop(pin);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FILLED, FILLING, FREE, NoSlot, PARTS, PartRange, SlotBook, advice_disjoint, audit_offsets,
        layer_bases, pick_slot, slots_of, window_bytes, window_span,
    };
    use bloomery_levers::NvTierEvict;
    use engram::direct::{AnonMap, DIRECT_ALIGN, aligned_span};
    use model::moe::Claim;
    use model::placement::{ARENA_LANE_SLOTS, UNION_BATCH_IDS};
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    fn book(state: u32, tick: u64) -> SlotBook {
        SlotBook {
            state: AtomicU32::new(state),
            id: AtomicU32::new(u32::MAX),
            tick: AtomicU64::new(tick),
            seq: AtomicU64::new(0),
        }
    }

    /// A part's window holds every aligned span its own byte range can ask
    /// for, at any skew the file can leave it at, and no window crosses a
    /// slot or another part's.
    #[test]
    fn a_parts_window_holds_its_aligned_spans() {
        let a = DIRECT_ALIGN;
        for len in [a / 2, a, 3 * a + 100, 921_600] {
            let w = window_bytes(len, 0);
            assert_eq!(w % a, 0, "{len}: the window is aligned");
            for skew in 0..a {
                // The part's bytes at `skew` into its last page before it.
                let at = 16 * a + skew;
                let (start, span, need) = aligned_span(at as u64, len as u64);
                let _ = need;
                assert!(
                    span <= w,
                    "{len} at skew {skew}: the span {span} passes the window {w}"
                );
                assert!(start % a as u64 == 0);
            }
        }
    }

    /// The floor of calls of `k` ids beside `lane_slots` lane slots over the
    /// slot set `sizes` sums to.
    fn floor_of(k: u64, lane_slots: u64, sizes: &[usize]) -> model::placement::ArenaFloor {
        let set = sizes.iter().sum::<usize>() as u64;
        model::placement::ArenaFloor::new(k, 1, lane_slots, set).expect("a floor")
    }

    /// The slots a budget holds: the same count a layer, the carve under
    /// the budget; a budget under one slot of every layer is the named
    /// refusal, and so is a plan that pages nothing.
    ///
    /// PIN(2026-10-12): the budgets are multiples of the floor's 37 slots,
    /// `2 × UNION_BATCH_IDS + LANE_THREADS + 1` (a prompt union's two batches
    /// and the lane's four open reads and victim claim, whatever the routed
    /// width), where they were 33 slots, and 10 and 1 before that:
    /// the floor raised, no refusal dropped.
    #[test]
    fn the_budget_carves_the_same_count_a_layer() {
        let sizes = [1_000_000, 1_200_000, 900_000];
        let total: u64 = sizes.iter().sum::<usize>() as u64;
        let one = floor_of(1, 0, &sizes);
        let floor = one.slots;
        assert_eq!(floor, 37);
        let n = slots_of((floor + 10) * total, &sizes, &one).expect("the floor and ten more");
        assert_eq!(n as u64, floor + 10);
        assert!(n as u64 * total <= (floor + 10) * total);
        assert_eq!(slots_of(floor * total, &sizes, &one), Ok(floor as usize));
        let under = slots_of(total - 1, &sizes, &one).unwrap_err();
        assert!(
            under.contains("carves 0 slots")
                && under.contains("under the floor of 37 slots")
                && under.contains(&(total - 1).to_string()),
            "{under}"
        );
        // A floor of no slot still refuses a carve of none.
        let none = floor_of(0, 0, &sizes);
        assert!(slots_of(total - 1, &sizes, &none).is_err());
        assert!(
            slots_of(total, &[], &one)
                .unwrap_err()
                .contains("no routed stack")
        );
    }

    /// A budget that carves one slot a layer fewer than the arena's floor —
    /// a prompt union's two batches and the lane's slots, thirty-seven slots,
    /// over ten ids a call and the lane's five slots — is refused by name,
    /// naming the carve, the floor's slots, its ids a call, its lane slots,
    /// the union's slots and its bytes over the paged layers; the floor's own
    /// budget carves exactly its slots.
    ///
    /// PIN(2026-10-12): the floor is 37 slots, not 33 (2026-10-11: 11): the
    /// batched union pins two batches while the next is filled and the lane
    /// holds `LANE_THREADS + 1` = 5 slots (2 × 16 + 5 > 10 × 1 + 5).
    #[test]
    fn a_carve_under_the_floor_is_refused_by_name() {
        let sizes = [3_133_440, 3_137_536, 3_133_440];
        let total: u64 = sizes.iter().sum::<usize>() as u64;
        let least = floor_of(10, ARENA_LANE_SLOTS, &sizes);
        assert_eq!(least.slots, 37);
        let under = slots_of(37 * total - 1, &sizes, &least).unwrap_err();
        for part in [
            "carves 36 slots a paged layer".to_string(),
            "under the floor of 37 slots".to_string(),
            "10 × 1 ids a call + 5 lane slots".to_string(),
            "37 for a union in batches".to_string(),
            format!("{} B over these layers", 37 * total),
        ] {
            assert!(under.contains(&part), "{part:?} in {under}");
        }
        assert!(slots_of(9 * total, &sizes, &least).is_err(), "k − 1 slots");
        assert_eq!(slots_of(37 * total, &sizes, &least), Ok(37));
    }

    /// The slot a miss takes: the first free one, else the least recently
    /// used filled one no lane's read pins; a filling slot is nobody's to
    /// take, and a miss with nothing left is refused by name, with the
    /// reason, never waited for.
    #[test]
    fn a_miss_takes_a_free_slot_else_the_least_recently_used_unpinned() {
        let open = |_: &SlotBook| false;
        let free = [book(FREE, 0), book(FILLED, 5), book(FILLED, 2)];
        assert_eq!(pick_slot(&free, open, None), Ok(0), "the first free slot");
        let lru = [book(FILLED, 5), book(FILLED, 2), book(FILLED, 9)];
        assert_eq!(
            pick_slot(&lru, open, None),
            Ok(1),
            "the least recently used"
        );
        let mixed = [book(FILLING, 1), book(FILLED, 7), book(FILLED, 3)];
        assert_eq!(
            pick_slot(&mixed, open, None),
            Ok(2),
            "a filling slot is not taken"
        );
        assert_eq!(
            pick_slot(&[book(FILLING, 1)], open, None),
            Err(NoSlot::Filling),
            "all filling: raced"
        );
        assert_eq!(
            pick_slot(&[], open, None),
            Err(NoSlot::Filling),
            "no slots at all"
        );
        // The least recently used slot (index 1) is pinned: the next oldest
        // goes, and the pinned one stays whatever its age.
        for (i, b) in lru.iter().enumerate() {
            b.id.store(i as u32, std::sync::atomic::Ordering::Relaxed);
        }
        let pins_one = |s: &SlotBook| s.id.load(std::sync::atomic::Ordering::Relaxed) == 1;
        assert_eq!(
            pick_slot(&lru, pins_one, None),
            Ok(0),
            "the oldest unpinned"
        );
        // A free slot is taken before any pin is read; every filled slot
        // pinned is the floor's refusal, not the race's.
        assert_eq!(
            pick_slot(&free, |_| true, None),
            Ok(0),
            "a free slot needs no unpinned one"
        );
        assert_eq!(
            pick_slot(&lru, |_| true, None),
            Err(NoSlot::Pinned),
            "all pinned: the named refusal"
        );
        let filling_and_pinned = [book(FILLING, 1), book(FILLED, 3)];
        assert_eq!(
            pick_slot(&filling_and_pinned, |_| true, None),
            Err(NoSlot::Pinned),
            "a filling slot beside the pinned one"
        );
    }

    /// Under heat a miss takes the oldest transient first, whatever its
    /// count, then the hot slot whose expert has the least count, the least
    /// recently used among equal counts; a pinned slot is passed over and a
    /// free one taken first, as under tick.
    #[test]
    fn a_heat_miss_takes_a_transient_then_the_coldest_count() {
        use std::sync::atomic::Ordering::Relaxed;
        let mut counts = super::LayerCounts::new(8);
        for (id, n) in [(0u32, 5.0), (1, 1.0), (2, 1.0), (3, 0.5), (4, 9.0)] {
            counts.add(id, n);
        }
        let heat = Some(&counts);
        // Each slot's id, tick and sequence word (a transient's from 1).
        let slots = |books: &[(u32, u64, u64)]| -> Vec<SlotBook> {
            books
                .iter()
                .map(|&(id, tick, seq)| {
                    let b = book(FILLED, tick);
                    b.id.store(id, Relaxed);
                    b.seq.store(seq, Relaxed);
                    b
                })
                .collect()
        };
        let open = |_: &SlotBook| false;
        let hot = slots(&[(0, 1, 0), (1, 7, 0), (2, 4, 0), (4, 9, 0)]);
        assert_eq!(
            pick_slot(&hot, open, heat),
            Ok(2),
            "the least count, the older of two"
        );
        assert_eq!(pick_slot(&hot, open, None), Ok(0), "tick: the oldest");
        let pins_two = |s: &SlotBook| s.id.load(Relaxed) == 2;
        assert_eq!(pick_slot(&hot, pins_two, heat), Ok(1), "the next coldest");
        let mixed = slots(&[(0, 1, 0), (4, 0, 2), (3, 9, 0), (2, 0, 1)]);
        assert_eq!(
            pick_slot(&mixed, open, heat),
            Ok(3),
            "the oldest transient before the coldest hot slot"
        );
        let free = [book(FILLED, 1), book(FREE, 0)];
        assert_eq!(pick_slot(&free, |_| true, heat), Ok(1), "a free slot");
        assert_eq!(
            pick_slot(&hot, |_| true, heat),
            Err(NoSlot::Pinned),
            "all pinned"
        );
    }

    /// A lane's open read pins its id's slot against the claim that would
    /// evict it, from the open to the end: the claim takes the next oldest
    /// slot, and with every filled slot pinned refuses by name, the slots
    /// and the books as they were. The mapping's reads and the slot's are
    /// told apart by the open count against the mapping's.
    #[test]
    fn an_open_read_keeps_its_slot_out_of_the_lru() {
        let arena = arena_of(4, 2);
        let mut books = super::Books::default();
        let first = super::claim_slots(0, &arena, &mut books, &[0, 1], Claim::Step).unwrap();
        fill_claimed(&arena, &mut books, &first);
        // Expert 0 is the least recently used and a lane has it open.
        arena.open(0, 0, "test").expect("the lane's open");
        let evict = super::claim_slots(0, &arena, &mut books, &[2], Claim::Step).unwrap();
        assert_eq!(evict.evicted.len(), 1);
        assert!(arena.filled(0).is_some(), "the pinned slot stays filled");
        assert!(arena.filled(1).is_none(), "the next oldest was taken");
        fill_claimed(&arena, &mut books, &evict);
        // Every filled slot pinned: refused by name, nothing moved.
        arena.open(0, 2, "test").expect("a second lane's open");
        let tick = books.tick;
        let e = super::claim_slots(0, &arena, &mut books, &[3], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read"), "{e}");
        assert_eq!(books.tick, tick, "the books' tick stands");
        assert!(arena.filled(0).is_some() && arena.filled(2).is_some());
        // The ends unpin: the slot is the LRU's again.
        arena.close(0, 0, "test").expect("the end");
        arena.close(0, 2, "test").expect("the end");
        let again = super::claim_slots(0, &arena, &mut books, &[3], Claim::Step).unwrap();
        assert_eq!(again.evicted.len(), 1, "an unpinned slot goes again");
    }

    /// `ids` claimed as `kind` and filled, as a call's claim and its fill's
    /// join do.
    fn ensure_as(
        arena: &super::LayerArena,
        books: &mut super::Books,
        ids: &[u32],
        kind: Claim,
    ) -> super::Claimed {
        let c = super::claim_slots(0, arena, books, ids, kind).unwrap();
        fill_claimed_as(arena, books, &c, kind);
        c
    }

    /// The ids the arena holds filled, ascending.
    fn filled_ids(arena: &super::LayerArena) -> Vec<u32> {
        (0..arena.n_expert)
            .filter(|&id| arena.filled(id).is_some())
            .collect()
    }

    #[test]
    fn a_tick_claim_keeps_hits_named_after_its_first_miss() {
        for kind in [Claim::Step, Claim::Keep, Claim::Transient] {
            let arena = arena_of(8, 2);
            let mut books = super::Books::default();
            ensure_as(&arena, &mut books, &[2, 7], Claim::Step);
            let claim = ensure_as(&arena, &mut books, &[1, 2], kind);
            assert_eq!(claim.misses, 1, "{kind:?}: the later hit stays filled");
            assert_eq!(filled_ids(&arena), [1, 2]);
        }
    }

    #[test]
    fn a_queued_drop_keeps_a_read_opened_after_the_request() {
        const LAYER: usize = 1;
        let scratch = Scratch::new("queued-read");
        let mut tier = card_tier(&scratch);
        tier.drops.queue().stop = true;
        tier.drops.work.notify_all();
        tier.dropper.lock().unwrap().take().unwrap().join().unwrap();
        let arena = std::sync::Arc::get_mut(&mut tier).unwrap().by_layer[LAYER]
            .as_mut()
            .unwrap();
        for (id, parts) in arena.parts.iter_mut().enumerate() {
            parts.fill(one_run(0, (id * CARD_PAGE) as u64, CARD_PAGE, 0));
        }
        tier.open_read(LAYER, 1).unwrap();
        tier.end_read(LAYER, 1).unwrap();
        tier.open_read(LAYER, 2).unwrap();
        let arena = tier.by_layer[LAYER].as_ref().unwrap();
        let want = tier.runs_of(arena, &arena.layer_marks(), "test").unwrap();
        assert_eq!(
            tier.drops.queue().pending[LAYER].as_ref(),
            Some(&want),
            "the queued ranges exclude every currently open read"
        );
        tier.end_read(LAYER, 2).unwrap();
    }

    #[test]
    fn an_active_dropper_finishes_before_a_new_read_opens() {
        active_drop_before_read(true);
    }

    #[test]
    fn an_active_drop_layer_finishes_before_a_new_read_opens() {
        active_drop_before_read(false);
    }

    fn active_drop_before_read(queued: bool) {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        const LAYER: usize = 1;
        let scratch = Scratch::new(if queued {
            "active-queued"
        } else {
            "active-layer"
        });
        let tier = card_tier(&scratch);
        let (entered_tx, entered) = mpsc::channel();
        let (resume, resume_rx) = mpsc::channel();
        *tier.drops.release_pause.lock().unwrap() = Some((entered_tx, resume_rx));
        let drop = if queued {
            tier.open_read(LAYER, 1).unwrap();
            tier.end_read(LAYER, 1).unwrap();
            None
        } else {
            let tier = Arc::clone(&tier);
            Some(std::thread::spawn(move || tier.drop_layer(LAYER).unwrap()))
        };
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        let (starting_tx, starting) = mpsc::channel();
        let (opened_tx, opened) = mpsc::channel();
        let reader = {
            let tier = Arc::clone(&tier);
            std::thread::spawn(move || {
                starting_tx.send(()).unwrap();
                tier.open_read(LAYER, 2).unwrap();
                opened_tx.send(()).unwrap();
                tier.end_read(LAYER, 2).unwrap();
            })
        };
        starting.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = opened.recv_timeout(Duration::from_millis(100));
        resume.send(()).unwrap();
        if let Some(drop) = drop {
            drop.join().unwrap();
        }
        reader.join().unwrap();
        assert_eq!(
            early,
            Err(mpsc::RecvTimeoutError::Timeout),
            "queued={queued}: a new read waits for the active page release"
        );
        opened.recv_timeout(Duration::from_secs(5)).unwrap();
        tier.flush().unwrap();
    }

    #[test]
    fn a_card_claim_and_its_host_tail_count_one_column() {
        use model::moe::TierSlots;
        let scratch = Scratch::new("card-host-heat");
        let tier = eight_tier(&scratch, 3, NvTierEvict::Heat);
        let ids = [1];
        let ticket = tier.issue_card_hold(1, &ids).unwrap();
        let host = TierSlots::prefetch_pinned(tier.as_ref(), 1, &[], Claim::Step, 0, None)
            .unwrap()
            .wait()
            .unwrap();
        drop(host);
        drop(ticket.finish().unwrap());
        assert_eq!(tier.route_count(1, 1), Some(1.0));
        drop(
            TierSlots::prefetch_pinned(tier.as_ref(), 1, &[2], Claim::Step, 0, None)
                .unwrap()
                .wait()
                .unwrap(),
        );
        assert_eq!(tier.route_count(1, 1), Some(1.0));
        assert_eq!(tier.route_count(1, 2), Some(1.0));
        drop(
            TierSlots::prefetch_pinned(tier.as_ref(), 1, &[], Claim::Step, 1, None)
                .unwrap()
                .wait()
                .unwrap(),
        );
        assert_eq!(
            tier.books.lock().unwrap().counts.as_ref().unwrap()[1].since,
            2,
            "the standalone empty column advances the clock once"
        );
    }

    #[test]
    fn heat_counts_keep_the_same_rank_across_prompt_splits() {
        heat_without_arena_ids("prompt");
    }

    #[test]
    fn heat_counts_keep_the_same_rank_across_step_splits() {
        heat_without_arena_ids("step");
    }

    #[test]
    fn heat_counts_keep_the_same_rank_across_resident_columns() {
        heat_without_arena_ids("resident");
    }

    #[test]
    fn heat_counts_keep_the_same_rank_across_card_and_host_legs() {
        heat_without_arena_ids("card-resident");
        heat_without_arena_ids("card-mapped");
    }

    fn heat_without_arena_ids(mode: &str) {
        use engram::direct::DirectFile;
        use gguf::write::{Layout, TensorDecl, Writer};
        use model::Tensor2;
        use model::moe::{HostLayer, HostLayerSpec, HostScratch, UnionScratch};
        use model::ops::Lanes;
        use model::placement::workstation::ArenaGrant;
        use model::r8file::R8Source;
        use std::sync::Arc;
        const WIDTH: usize = 256;
        const SLOTS: usize = 2 * UNION_BATCH_IDS as usize + 2;
        const EXPERTS: usize = SLOTS + 2;
        let scratch = Scratch::new(&format!("heat-{mode}"));
        let path = scratch.0.join("layer.gguf");
        let names = ["gate_exps", "up_exps", "down_exps"];
        let ty = gguf::GgmlType::Q4_K;
        let per =
            WIDTH * WIDTH / ty.blck_size().unwrap() as usize * ty.type_size().unwrap() as usize;
        let bytes = vec![0; per * EXPERTS];
        let decls = names.map(|name| TensorDecl {
            name: name.into(),
            dims: vec![WIDTH as u64, WIDTH as u64, EXPERTS as u64],
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        });
        let layout = Layout::new(&[], decls.to_vec()).unwrap();
        let mut writer = Writer::new(std::fs::File::create(&path).unwrap(), layout).unwrap();
        for name in names {
            writer.tensor(name, &bytes).unwrap();
        }
        writer.finish().unwrap();
        let split = Arc::new(gguf::Split::open(&path).unwrap());
        let file = Arc::new(DirectFile::open_buffered(&path).unwrap());
        let source = R8Source::rows(&split);
        let spec = HostLayerSpec {
            gate: names[0],
            up: names[1],
            down: names[2],
            n_expert: EXPERTS,
            embd: WIDTH,
            ff: WIDTH,
            swiglu_limit: 0.0,
        };
        // The middle columns name no arena expert: the host segment's
        // expert alone, or no host expert at all (every expert on the card).
        let host: &[(u32, f32)] = &[(0, 1.0)];
        let expected = [2f64.powf(-130.0 / 32.0) + 2f64.powf(-129.0 / 32.0), 1.0];
        for (middle, mode) in [host, &[]]
            .into_iter()
            .flat_map(|middle| [(middle, "whole"), (middle, mode)])
        {
            let routes: Vec<&[(u32, f32)]> = (0..131)
                .map(|col| match col {
                    0 | 1 => &[(1, 1.0)][..],
                    130 => &[(2, 1.0)][..],
                    _ => middle,
                })
                .collect();
            let mut arena = arena_of(EXPERTS as u32, 0);
            arena.host = 0..1;
            arena.windows = std::array::from_fn(|part| part * window_bytes(per, 0));
            arena.slot_bytes = PARTS * window_bytes(per, 0);
            arena.names = names.map(str::to_owned);
            arena.files = shards_of([Arc::clone(&file)]);
            if !mode.starts_with("card-") {
                arena.part_of.set([0, 1, 2]).unwrap();
            }
            arena.parts = (0..EXPERTS)
                .map(|id| {
                    std::array::from_fn(|part| {
                        let (shard, info) = split.find(names[part]).unwrap();
                        let at = split.shard(shard).unwrap().data_base()
                            + info.offset
                            + (id * per) as u64;
                        one_run(0, at, per, at as usize % DIRECT_ALIGN)
                    })
                })
                .collect();
            let slot_bytes = arena.slot_bytes;
            let mut tier = super::NvTier::carve(
                vec![Some(arena)],
                &[slot_bytes],
                SLOTS,
                ArenaGrant::whole((SLOTS * slot_bytes) as u64),
                0,
                &split,
                super::ArenaKind::Anon,
            )
            .unwrap();
            tier.count_routes();
            let tier = Arc::new(tier);
            let mut layer = HostLayer::build(source, &spec).unwrap();
            layer.attach_tier(0, 0..1, tier.clone());
            let mut union = UnionScratch::new_routed(WIDTH, WIDTH, routes.len(), 1).unwrap();
            let mut host = HostScratch::new(WIDTH, WIDTH, 1).unwrap();
            let mut run = if mode.starts_with("card-") {
                let mut run = super::super::run::HostRun::build(
                    Arc::clone(&split),
                    0,
                    false,
                    super::super::run::HostWidths {
                        embd: WIDTH,
                        ff: WIDTH,
                        n_used: 1,
                    },
                    |src| HostLayer::build(src, &spec).map(|layer| vec![layer]),
                )
                .unwrap();
                run.attach_tier(Some(Arc::clone(&tier))).unwrap();
                if mode == "card-resident" {
                    run.prepare_union(1).unwrap();
                }
                Some(run)
            } else {
                None
            };
            tier.ensure_as(0, &(1..=SLOTS as u32).collect::<Vec<_>>(), Claim::Keep)
                .unwrap();
            let chunks = match mode {
                "whole" => vec![131],
                "prompt" | "step" => vec![2, 128, 1],
                _ => vec![1; 131],
            };
            let mut start = 0;
            for cols in chunks {
                let x = Tensor2::zeros(WIDTH, cols);
                let mut out = vec![f32::NAN; WIDTH * cols];
                let lists = &routes[start..start + cols];
                let card_id = [lists[0].first().map_or(0, |&(id, _)| id)];
                let ticket = match mode {
                    "card-resident" | "card-mapped" if card_id[0] != 0 => {
                        Some(tier.issue_card_hold(0, &card_id).unwrap())
                    }
                    _ => None,
                };
                let empty: [&[(u32, f32)]; 1] = [&[]];
                let lists = if ticket.is_some() { &empty } else { lists };
                match mode {
                    "whole" | "prompt" => {
                        layer.experts_union_into(source, &x, lists, &mut out, &mut union)
                    }
                    "step" => layer.experts_step_union_into_on(
                        Lanes::Flat,
                        source,
                        &x,
                        lists,
                        &mut out,
                        &mut union,
                    ),
                    "card-resident" | "card-mapped" => {
                        use super::super::HostExperts;
                        run.as_mut()
                            .unwrap()
                            .experts_into(0, &x, lists[0], &mut out, u64::from(ticket.is_none()))
                            .unwrap();
                        Ok(())
                    }
                    _ => layer.experts_into_on(
                        Lanes::Flat,
                        source,
                        &x,
                        lists[0],
                        &mut out,
                        &mut host,
                        Some(&mut union),
                        1,
                    ),
                }
                .unwrap();
                if let Some(ticket) = ticket {
                    drop(ticket.finish().unwrap());
                }
                assert!(
                    out.iter().all(|&v| v == 0.0),
                    "{mode} {middle:?}: the zero weights"
                );
                start += cols;
            }
            for (id, want) in [1, 2].into_iter().zip(expected) {
                let got = tier.route_count(0, id).unwrap();
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "{mode} {middle:?}: expert {id}: {got} != {want}"
                );
            }
            let protected = tier
                .pin(0, &(3..=SLOTS as u32).collect::<Vec<_>>())
                .unwrap();
            tier.ensure_as(0, &[(SLOTS + 1) as u32], Claim::Keep)
                .unwrap();
            assert!(
                tier.by_layer[0].as_ref().unwrap().filled(1).is_none(),
                "{mode} {middle:?}: the colder expert leaves"
            );
            assert!(
                tier.by_layer[0].as_ref().unwrap().filled(2).is_some(),
                "{mode} {middle:?}: the hotter expert stays"
            );
            drop(protected);
        }
    }

    /// A transient miss enters at the cold end and is the first victim,
    /// whatever claim takes the next slot: transients go oldest first (the
    /// order their call claimed them in), and only then the least recently
    /// used hot slot.
    #[test]
    fn a_transient_claim_is_the_first_victim() {
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1], Claim::Step);
        // Expert 3 is claimed before 2, so it is the older transient.
        ensure_as(&arena, &mut books, &[3, 2], Claim::Transient);
        assert_eq!(filled_ids(&arena), [0, 1, 2, 3]);
        // A transient claim's miss takes the oldest transient too.
        ensure_as(&arena, &mut books, &[4], Claim::Transient);
        assert_eq!(filled_ids(&arena), [0, 1, 2, 4], "3 was the oldest");
        // Expert 0 is the least recently used, and still outlives them all.
        ensure_as(&arena, &mut books, &[5], Claim::Step);
        assert_eq!(filled_ids(&arena), [0, 1, 4, 5], "2 was the next oldest");
        ensure_as(&arena, &mut books, &[6], Claim::Step);
        assert_eq!(filled_ids(&arena), [0, 1, 5, 6], "the last transient");
        ensure_as(&arena, &mut books, &[7], Claim::Step);
        assert_eq!(filled_ids(&arena), [1, 5, 6, 7], "no transient: the LRU");
    }

    /// A keep claim takes a fresh tick like a step, so every transient goes
    /// before it, though it is older than the hot slots beside it; with no
    /// transient left it is the least recently used.
    #[test]
    fn a_keep_claim_outlives_every_transient() {
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[1], Claim::Keep);
        ensure_as(&arena, &mut books, &[0, 2], Claim::Transient);
        ensure_as(&arena, &mut books, &[3], Claim::Step);
        ensure_as(&arena, &mut books, &[4], Claim::Keep);
        assert_eq!(
            filled_ids(&arena),
            [1, 2, 3, 4],
            "0 was the oldest transient"
        );
        ensure_as(&arena, &mut books, &[5], Claim::Step);
        assert_eq!(filled_ids(&arena), [1, 3, 4, 5], "2 was the last transient");
        ensure_as(&arena, &mut books, &[6], Claim::Step);
        assert_eq!(
            filled_ids(&arena),
            [3, 4, 5, 6],
            "the kept id is the LRU now"
        );
    }

    /// A step or keep hit on a transient gives it a fresh tick and turns it
    /// hot: it leaves the cold end, and is the least recently used only by
    /// its new tick.
    #[test]
    fn a_step_hit_turns_a_transient_hot() {
        for hit in [Claim::Step, Claim::Keep] {
            let arena = arena_of(8, 3);
            let mut books = super::Books::default();
            ensure_as(&arena, &mut books, &[0], Claim::Step);
            ensure_as(&arena, &mut books, &[1, 2], Claim::Transient);
            let c = ensure_as(&arena, &mut books, &[1], hit);
            assert_eq!(c.misses, 0, "{hit:?}: a hit");
            // Expert 2 is the only transient left.
            ensure_as(&arena, &mut books, &[3], Claim::Step);
            assert_eq!(filled_ids(&arena), [0, 1, 3], "{hit:?}: 1 is hot, 2 went");
            // Expert 1's tick is newer than 0's: 0 is the LRU.
            ensure_as(&arena, &mut books, &[4], Claim::Step);
            assert_eq!(filled_ids(&arena), [1, 3, 4], "{hit:?}: the LRU is 0");
        }
    }

    /// A transient hit never demotes a hot id and leaves it where it was in
    /// the order of use (it is still the LRU), and leaves a transient where
    /// it was, the first victim.
    ///
    /// PIN(2026-10-11): the hot id's last assertion was "refreshed after 1: 1
    /// is the LRU now"; derived: a refresh puts an id a call merely named
    /// above the keep ids claimed before it, so once the transients a batch
    /// frees fall short the least recently used hot slot, a keep id, goes in
    /// its place (`a_transient_hit_leaves_a_hot_id_s_tick`).
    #[test]
    fn a_transient_hit_never_demotes_a_hot_id_nor_warms_a_transient() {
        let arena = arena_of(8, 3);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1], Claim::Step);
        ensure_as(&arena, &mut books, &[2], Claim::Transient);
        let c = ensure_as(&arena, &mut books, &[0, 2], Claim::Transient);
        assert_eq!(c.misses, 0, "both hit");
        ensure_as(&arena, &mut books, &[3], Claim::Step);
        assert_eq!(filled_ids(&arena), [0, 1, 3], "2 is still the first victim");
        // Expert 0 keeps its tick, older than 1's: it is the LRU still.
        ensure_as(&arena, &mut books, &[4], Claim::Step);
        assert_eq!(filled_ids(&arena), [1, 3, 4], "the hot hit stayed the LRU");
    }

    /// A transient claim's hit on a hot id leaves its tick alone, a step's or
    /// a keep's hit on it takes a fresh one: the hot id the transient call
    /// merely named stays older than the keep ids claimed before it, so the
    /// misses that follow take it, not a keep id.
    #[test]
    fn a_transient_hit_leaves_a_hot_id_s_tick() {
        let tick = |arena: &super::LayerArena, id: u32| {
            let slot = arena.filled(id).expect("filled");
            arena.slots[slot]
                .tick
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        let arena = arena_of(8, 3);
        let mut books = super::Books::default();
        // 0 is hot first (a spare), then the call's keep claim takes 1, 2.
        ensure_as(&arena, &mut books, &[0], Claim::Step);
        ensure_as(&arena, &mut books, &[1, 2], Claim::Keep);
        let before = tick(&arena, 0);
        let c = ensure_as(&arena, &mut books, &[0], Claim::Transient);
        assert_eq!(c.misses, 0, "a hit");
        assert_eq!(tick(&arena, 0), before, "the transient hit left the tick");
        assert!(before < tick(&arena, 1), "older than the keep ids");
        // A miss takes 0, the oldest, never a keep id.
        ensure_as(&arena, &mut books, &[3], Claim::Transient);
        assert_eq!(filled_ids(&arena), [1, 2, 3], "0 went, the keep ids stayed");
        // A step or keep hit still refreshes.
        let arena = arena_of(8, 3);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1, 2], Claim::Step);
        for kind in [Claim::Step, Claim::Keep] {
            let before = tick(&arena, 0);
            ensure_as(&arena, &mut books, &[0], kind);
            assert!(tick(&arena, 0) > before, "{kind:?}: a fresh tick");
        }
    }

    /// A transient call's own ids are never its victims, whatever order it
    /// names them in: a transient it hits keeps its cold place, where its
    /// own misses would look first.
    #[test]
    fn a_transient_call_never_evicts_its_own_ids() {
        for names in [[0, 3], [3, 0]] {
            let arena = arena_of(8, 3);
            let mut books = super::Books::default();
            ensure_as(&arena, &mut books, &[2], Claim::Step);
            ensure_as(&arena, &mut books, &[0, 1], Claim::Transient);
            // Expert 0 is the oldest transient and the call names it.
            let c = ensure_as(&arena, &mut books, &names, Claim::Transient);
            assert_eq!((c.misses, c.evicted.len()), (1, 1), "{names:?}");
            assert_eq!(filled_ids(&arena), [0, 2, 3], "{names:?}: 1 went, not 0");
        }
    }

    /// The union's counters: a pinned claim's ids go by kind, its hits to
    /// `union_hits` under keep and transient and to `step_hits` under step,
    /// a pinned step claim to neither kind; the calls by path.
    #[test]
    fn the_union_counters_follow_the_claim_kind() {
        let s = super::Stats::default();
        s.pinned_claim(Claim::Keep, 3, 5);
        s.pinned_claim(Claim::Transient, 1, 2);
        s.pinned_claim(Claim::Step, 4, 1);
        s.union_call(true);
        s.union_call(true);
        s.union_call(false);
        let load = |a: &AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(load(&s.keep_claims), 8, "3 hits + 5 misses");
        assert_eq!(load(&s.transient_claims), 3, "1 hit + 2 misses");
        assert_eq!(load(&s.union_hits), 4, "3 + 1: a step claim's 4 are not");
        assert_eq!(load(&s.step_hits), 4, "the step claim's 4");
        assert_eq!(
            (load(&s.union_fills), load(&s.step_fills)),
            (0, 0),
            "the fill site counts the fills"
        );
        assert_eq!((load(&s.union_batched), load(&s.union_small)), (2, 1));
        s.note_step_miss(Claim::Keep);
        s.note_step_miss(Claim::Step);
        s.note_step_miss(Claim::Step);
        s.note_step_wall(Claim::Step, 9);
        s.note_step_wall(Claim::Transient, 4);
        s.note_hits_under_read(7);
        assert_eq!(
            load(&s.step_miss_layers),
            2,
            "a keep claim is not a step layer"
        );
        assert_eq!(
            load(&s.step_fill_wall_ns),
            9,
            "a transient wall is not a step's"
        );
        assert_eq!(load(&s.step_hits_under_read_ns), 7);
    }

    /// A union book holds the arena's ids of the call with their columns,
    /// ascending: the plan's host segment is the mapping's and is left out.
    #[test]
    fn a_union_book_leaves_out_the_host_segment() {
        // Ids 1, 2, 3, 6 listed by 2, 1, 4 and 3 columns; 2..4 is the host's.
        let book = super::UnionBook::of(&(2..4), &[1, 2, 3, 6], &[0, 2, 3, 7, 10], true);
        assert_eq!(book.cols, [(1, 2), (6, 3)]);
        assert!(book.batched);
    }

    /// A slot's heat: a step or keep claim's is hot, a transient's is not,
    /// a hit of a keep on a transient turns it hot, and an id with no slot
    /// has none.
    #[test]
    fn a_slots_heat_follows_its_claim() {
        use super::Heat;
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0], Claim::Step);
        ensure_as(&arena, &mut books, &[1], Claim::Keep);
        ensure_as(&arena, &mut books, &[2, 3], Claim::Transient);
        let heat = |ids: &[u32]| ids.iter().map(|&id| arena.heat(id)).collect::<Vec<_>>();
        assert_eq!(
            heat(&[0, 1, 2, 3, 4]),
            [
                Some(Heat::Hot),
                Some(Heat::Hot),
                Some(Heat::Transient),
                Some(Heat::Transient),
                None
            ]
        );
        ensure_as(&arena, &mut books, &[3], Claim::Keep);
        assert_eq!(heat(&[2, 3]), [Some(Heat::Transient), Some(Heat::Hot)]);
    }

    /// A pinned batch is no victim, hot or transient, whatever its age: the
    /// next oldest unpinned slot goes. A lane's open read and a batch pin
    /// count one each, so dropping the guard leaves the lane's read pinning.
    #[test]
    fn a_pinned_batch_is_never_a_victim() {
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1, 2, 3], Claim::Step);
        // Experts 0 and 1 are the two least recently used.
        let pin = arena.pin_batch(0, &[1, 0, 1], "test").expect("a batch pin");
        ensure_as(&arena, &mut books, &[4], Claim::Step);
        ensure_as(&arena, &mut books, &[5], Claim::Step);
        assert_eq!(
            filled_ids(&arena),
            [0, 1, 4, 5],
            "2 and 3 went, not 0 and 1"
        );
        drop(pin);
        ensure_as(&arena, &mut books, &[6], Claim::Step);
        assert_eq!(
            filled_ids(&arena),
            [1, 4, 5, 6],
            "the guard's drop: 0 is the LRU"
        );

        // A pinned transient outranks the transient-first rule.
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0], Claim::Step);
        ensure_as(&arena, &mut books, &[1, 2, 3], Claim::Transient);
        let pin = arena.pin_batch(0, &[1], "test").expect("a batch pin");
        ensure_as(&arena, &mut books, &[4], Claim::Step);
        assert_eq!(filled_ids(&arena), [0, 1, 3, 4], "2 went, not the pinned 1");
        // A lane's read on the same id keeps it pinned past the guard.
        arena.open(0, 1, "test").expect("the lane's open");
        drop(pin);
        ensure_as(&arena, &mut books, &[5], Claim::Step);
        assert_eq!(filled_ids(&arena), [0, 1, 4, 5], "3 went: the lane pins 1");
        arena.close(0, 1, "test").expect("the lane's end");
        ensure_as(&arena, &mut books, &[6], Claim::Transient);
        assert_eq!(
            filled_ids(&arena),
            [0, 4, 5, 6],
            "1 is the oldest transient"
        );
    }

    /// With every filled slot pinned a claim is refused by name, whatever
    /// its kind, and gives back what it had taken: a call whose first miss
    /// takes the one unpinned slot and whose second finds none leaves no
    /// slot filling, the evicted id filled again and the books as they were.
    #[test]
    fn every_slot_pinned_is_refused_by_name() {
        let arena = arena_of(8, 3);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1, 2], Claim::Step);
        let pin = arena.pin_batch(0, &[0, 1, 2], "test").expect("a batch pin");
        let tick = books.tick;
        for kind in [Claim::Step, Claim::Keep, Claim::Transient] {
            let e = super::claim_slots(0, &arena, &mut books, &[3], kind)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("pinned by a lane's read or a batch pin"),
                "{kind:?}: {e}"
            );
        }
        assert_eq!(books.tick, tick, "the books' tick stands");
        assert_eq!(filled_ids(&arena), [0, 1, 2]);
        drop(pin);

        // One unpinned slot, two misses: the first takes it, the second is
        // refused, and the first's slot is given back.
        let pin = arena.pin_batch(0, &[0, 1], "test").expect("a batch pin");
        let e = super::claim_slots(0, &arena, &mut books, &[3, 4], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        let filling = arena
            .slots
            .iter()
            .filter(|s| s.state.load(std::sync::atomic::Ordering::Relaxed) == FILLING)
            .count();
        assert_eq!(filling, 0, "no slot is left filling");
        assert_eq!(
            filled_ids(&arena),
            [0, 1, 2],
            "the evicted id is filled again"
        );
        assert!(arena.filled(3).is_none() && arena.filled(4).is_none());
        // The slot is claimable again: one miss fits.
        let c = super::claim_slots(0, &arena, &mut books, &[3], Claim::Step).unwrap();
        assert_eq!((c.picks.len(), c.evicted.len()), (1, 1));
        drop(pin);
    }

    /// A layer of `slots` slots under a prompt union's two pinned batches
    /// of `UNION_BATCH_IDS` ids, the lane's threads each pinning the slot of
    /// the id they stage (a victim filled unpinned, then opened) and one
    /// more victim claim beside them: `Err` with the refusal that claim or
    /// fill met.
    fn union_beside_lane(slots: usize, lane_threads: u32) -> Result<(), String> {
        let batch = UNION_BATCH_IDS as u32;
        let arena = arena_of(2 * batch + lane_threads + 1, slots);
        let mut books = super::Books::default();
        let mut pins = Vec::new();
        for b in 0..2 {
            let ids: Vec<u32> = (b * batch..(b + 1) * batch).collect();
            ensure_as(&arena, &mut books, &ids, Claim::Step);
            pins.push(arena.pin_batch(0, &ids, "test").expect("a batch pin"));
        }
        for id in 2 * batch..2 * batch + lane_threads + 1 {
            let c = super::claim_slots(0, &arena, &mut books, &[id], Claim::Step)
                .map_err(|e| e.to_string())?;
            fill_claimed(&arena, &mut books, &c);
            if id < 2 * batch + lane_threads {
                arena.open(0, id, "test").expect("the lane's open");
            }
        }
        Ok(())
    }

    /// The floor holds a union beside a lane: two pinned batches, four lane
    /// threads' open reads and one victim claim need the floor's slots, and
    /// a layer one slot under it refuses that claim by name.
    ///
    /// PIN(2026-10-12): the floor's slots are `2 × UNION_BATCH_IDS +
    /// LANE_THREADS + 1` = 37; the lane's one-slot term (33 slots) refused
    /// the claim beside the fourth read.
    #[test]
    fn the_floor_holds_a_union_beside_a_lanes_reads_and_claim() {
        let lane_threads = 4;
        let floor = floor_of(10, ARENA_LANE_SLOTS, &[1]).slots as usize;
        assert_eq!(
            floor,
            2 * UNION_BATCH_IDS as usize + lane_threads as usize + 1
        );
        union_beside_lane(floor, lane_threads).expect("the floor's slots hold them all");
        let e = union_beside_lane(floor - 1, lane_threads).unwrap_err();
        assert!(
            e.contains("pinned by a lane's read or a batch pin"),
            "one slot under the floor: {e}"
        );
    }

    /// The open reads of experts `ids` as the books count them.
    fn open_reads(arena: &super::LayerArena, ids: &[u32]) -> Vec<u32> {
        ids.iter()
            .map(|&id| {
                arena
                    .reading(0, id, "test")
                    .expect("an id of the layer")
                    .load(std::sync::atomic::Ordering::Acquire)
            })
            .collect()
    }

    /// An `ensure_pinned` pins a hit as its claim finds it and a miss as its
    /// fill publishes, each under the books' lock of the claim or the
    /// publish: a claim that runs between the two — the lane's, or the next
    /// batch's — takes none of the batch's slots, the hit's pinned already
    /// and the miss's filling. Once the fill publishes, the miss is filled
    /// and pinned in the same hold, so no claim finds it filled and free. The
    /// guard's drop unpins the whole batch.
    #[test]
    fn ensure_pinned_pins_hits_and_picks_in_one_hold() {
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1], Claim::Step);
        // Hit 0 and misses 2 and 3: one slot, expert 1's, is left to take.
        let mut claim =
            super::claim_slots_pinned(0, &arena, &mut books, &[0, 2, 3], Claim::Keep).unwrap();
        assert_eq!(claim.pinned, [0], "the hit is pinned by its claim");
        assert_eq!(open_reads(&arena, &[0, 1, 2, 3]), [1, 0, 0, 0]);
        let mut held = super::BatchPin {
            arena: &arena,
            layer: 0,
            ids: std::mem::take(&mut claim.pinned),
        };
        // A claim between the claim and the publish: the hit's slot is
        // pinned and the misses' are filling, so it can take expert 1's one
        // slot and no more; the second miss is refused by name and the
        // first's slot is given back.
        let e = super::claim_slots(0, &arena, &mut books, &[4, 5], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        assert_eq!(filled_ids(&arena), [0, 1], "nothing of the batch was taken");
        assert!(arena.filled(2).is_none() && arena.filled(3).is_none());
        // The fill publishes: the misses are filled and pinned together.
        super::publish_pinned(&arena, &mut books, &claim.picks, Claim::Keep, &mut held).unwrap();
        assert_eq!(filled_ids(&arena), [0, 1, 2, 3]);
        assert_eq!(open_reads(&arena, &[0, 1, 2, 3]), [1, 0, 1, 1]);
        // Only expert 1 can be taken: a claim of two misses is refused whole.
        let e = super::claim_slots(0, &arena, &mut books, &[4, 5], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        assert_eq!(
            filled_ids(&arena),
            [0, 1, 2, 3],
            "no slot of the batch taken"
        );
        let taken = super::claim_slots(0, &arena, &mut books, &[4], Claim::Step).unwrap();
        assert_eq!(taken.evicted.len(), 1);
        fill_claimed(&arena, &mut books, &taken);
        assert_eq!(filled_ids(&arena), [0, 2, 3, 4], "1 went, the batch stayed");
        drop(held);
        assert_eq!(open_reads(&arena, &[0, 1, 2, 3, 4]), [0; 5]);
        ensure_as(&arena, &mut books, &[5, 6, 7], Claim::Step);
        assert_eq!(
            filled_ids(&arena),
            [4, 5, 6, 7],
            "the unpinned batch is evictable"
        );
    }

    /// A pinned claim that is refused closes the pins it had taken on its
    /// hits, whatever the refusal: no count outlives it.
    #[test]
    fn a_refused_pinned_claim_closes_its_hit_pins() {
        let arena = arena_of(8, 3);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1, 2], Claim::Step);
        // Lanes read 1 and 2; the claim pins its hit 0 and finds no slot for
        // 3.
        arena.open(0, 1, "test").unwrap();
        arena.open(0, 2, "test").unwrap();
        let e = super::claim_slots_pinned(0, &arena, &mut books, &[0, 3], Claim::Keep)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        assert_eq!(
            open_reads(&arena, &[0, 1, 2]),
            [0, 1, 1],
            "the hit's pin closed"
        );
        assert_eq!(filled_ids(&arena), [0, 1, 2]);
    }

    /// A pin over the filled `ids` of layer 0 as the seam holds it: claimed
    /// pinned, then its ids moved into a [`TierPin`] over the arena.
    fn pinned<'a>(
        arena: &'a super::LayerArena,
        books: &mut super::Books,
        ids: &[u32],
    ) -> model::moe::TierPin<'a> {
        let claim = super::claim_slots_pinned(0, arena, books, ids, Claim::Keep).unwrap();
        let held = super::BatchPin {
            arena,
            layer: 0,
            ids: claim.pinned,
        };
        model::moe::TierPin::new(arena, held.into_token())
    }

    /// A pin's id buffer comes from the layer's pool and goes back to it when
    /// the guard drops, so the next pin of the layer takes the same buffer
    /// and the steady state makes no allocator call for it. The most pins a
    /// layer holds at once (each row's card hold on a pass of
    /// [`super::super::page::MAX_ROWS`] rows, a step's pin and two union
    /// batches) take pooled buffers only. A pin past the pool's
    /// [`super::PIN_BUFFERS`] takes a fresh buffer, never a refusal, and the
    /// buffer joins the pool when its pin drops.
    #[test]
    fn a_pins_id_buffer_returns_to_the_layers_pool() {
        let arena = arena_of(8, 4);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1, 2], Claim::Step);
        let first = pinned(&arena, &mut books, &[0, 1, 2]);
        let at = first.ids().as_ptr();
        assert_eq!(first.ids(), [0, 1, 2]);
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [1, 1, 1]);
        drop(first);
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [0, 0, 0], "the drop unpins");
        let second = pinned(&arena, &mut books, &[0, 1, 2]);
        assert_eq!(
            second.ids().as_ptr(),
            at,
            "the second pin of the layer took the first's buffer"
        );
        drop(second);
        let idle = |arena: &super::LayerArena| arena.idle.lock().unwrap().len();
        assert_eq!(idle(&arena), super::PIN_BUFFERS);
        let pooled: Vec<*const u32> = arena
            .idle
            .lock()
            .unwrap()
            .iter()
            .map(|b| b.as_ptr())
            .collect();
        let most: Vec<_> = (0..super::super::page::MAX_ROWS + 3)
            .map(|_| pinned(&arena, &mut books, &[0, 1, 2]))
            .collect();
        assert!(
            most.iter().all(|p| pooled.contains(&p.ids().as_ptr())),
            "the most pins a layer holds at once took a fresh buffer"
        );
        drop(most);
        // One past the pool at once: every pooled buffer, then a fresh one.
        let held: Vec<_> = (0..super::PIN_BUFFERS + 1)
            .map(|_| pinned(&arena, &mut books, &[0, 1, 2]))
            .collect();
        assert_eq!(idle(&arena), 0);
        let n = super::PIN_BUFFERS as u32 + 1;
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [n, n, n]);
        drop(held);
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [0, 0, 0]);
        assert_eq!(
            idle(&arena),
            super::PIN_BUFFERS + 1,
            "the last pin's fresh buffer joined the pool"
        );
    }

    /// A prefetch ticket returns the pin only once the batch's fill has
    /// published: before the wait the batch's slots are claimed (no claim
    /// takes them) and its misses unfilled, and what the wait returns holds
    /// every id of the batch — the hit pinned at the claim, the misses at the
    /// publish — filled. The ticket owns the fill's completion: the closure
    /// it is made of runs the publish, as a lane thread's join would.
    #[test]
    fn a_prefetch_ticket_waits_to_the_pin() {
        use model::moe::{Ticket, TierPin};
        let arena = arena_of(8, 4);
        let books = std::sync::Mutex::new(super::Books::default());
        ensure_as(&arena, &mut books.lock().unwrap(), &[0], Claim::Step);
        let mut claim = {
            let mut b = books.lock().unwrap();
            super::claim_slots_pinned(0, &arena, &mut b, &[0, 1, 2], Claim::Keep).unwrap()
        };
        let mut held = super::BatchPin {
            arena: &arena,
            layer: 0,
            ids: std::mem::take(&mut claim.pinned),
        };
        let picks = claim.picks;
        let ticket = Ticket::new(|| {
            let mut b = books.lock().unwrap();
            super::publish_pinned(&arena, &mut b, &picks, Claim::Keep, &mut held).unwrap();
            drop(b);
            Ok(TierPin::new(&arena, held.into_token()))
        });
        // Before the wait: the misses are claimed, not filled, and a claim
        // past the batch takes the one free slot and no more.
        assert!(arena.filled(1).is_none() && arena.filled(2).is_none());
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [1, 0, 0]);
        let e = super::claim_slots(0, &arena, &mut books.lock().unwrap(), &[3, 4], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        assert!(arena.filled(3).is_none(), "the refused claim took nothing");
        let pin = ticket.wait().expect("the fill ended");
        assert_eq!(filled_ids(&arena), [0, 1, 2], "every id filled at the wait");
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [1, 1, 1], "and pinned");
        drop(pin);
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [0, 0, 0]);
    }

    /// A claim of `ids` as a prefetch takes it — the hits pinned, the misses
    /// filling — whose misses' reads are queued on `reader` as `read` says.
    fn queued<'a>(
        arena: &'a super::LayerArena,
        books: &'a std::sync::Mutex<super::Books>,
        stats: &'a super::Stats,
        reader: &'a super::Reader,
        ids: &[u32],
        kind: Claim,
        read: impl Fn(usize, usize) -> Result<(), crate::GpuError> + Send + Sync + 'static,
    ) -> super::Inflight<'a> {
        let claim = {
            let mut b = books.lock().unwrap();
            super::claim_slots_pinned(0, arena, &mut b, ids, kind).unwrap()
        };
        let held = super::BatchPin {
            arena,
            layer: 0,
            ids: claim.pinned,
        };
        let tier = super::TierParts {
            arena,
            books,
            stats,
            reader,
        };
        super::Inflight::queue(tier, held, claim.picks, kind, read).unwrap()
    }

    /// A prompt batch's fill runs on the reader's threads, not the caller's:
    /// the caller has its claim back while every read still holds, the
    /// misses claimed and unfilled; the join publishes them, pinned.
    #[test]
    fn a_prefetch_fills_off_the_caller() {
        use std::sync::{Mutex, mpsc};
        let arena = arena_of(8, 4);
        let books = Mutex::new(super::Books::default());
        let stats = super::Stats::default();
        let reader = super::Reader::start(2, "test-fill").unwrap();
        let (started_tx, started) = mpsc::channel();
        let (release_tx, release) = mpsc::channel::<()>();
        let release = Mutex::new(release);
        let inflight = queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[0, 1],
            Claim::Keep,
            move |_, _| {
                started_tx.send(std::thread::current().id()).unwrap();
                // Bounded, so a fill run on the caller (which never releases
                // itself) fails the test rather than hanging it.
                let released = release
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(5));
                released.expect("the test released the read");
                Ok(())
            },
        );
        // The in-flight parts run, each on a thread that is not this one, and
        // hold: the caller is back with the claim in the books and nothing
        // filled. Two threads, so two parts are in flight; the rest wait.
        let units = 2 * PARTS;
        let ran: Vec<_> = (0..2).map(|_| started.recv().unwrap()).collect();
        assert!(ran.iter().all(|&t| t != std::thread::current().id()));
        assert!(arena.filled(0).is_none() && arena.filled(1).is_none());
        assert_eq!(
            open_reads(&arena, &[0, 1]),
            [0, 0],
            "pinned as they publish"
        );
        for _ in 0..units {
            release_tx.send(()).unwrap();
        }
        let pin = inflight.finish().ok().expect("the fill ended");
        assert_eq!(filled_ids(&arena), [0, 1]);
        assert_eq!(
            open_reads(&arena, &[0, 1]),
            [1, 1],
            "and pinned at the join"
        );
        let counted = stats.fills.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(counted, 2);
        assert_eq!(
            {
                let r = reader.stats();
                (
                    (r.prompt.jobs, r.prompt.units),
                    (r.victim.jobs, r.victim.units),
                )
            },
            ((1, 2 * PARTS as u64), (0, 0)),
            "the reader ran the batch as a prompt fill, three parts a pick"
        );
        drop(pin);
        assert_eq!(open_reads(&arena, &[0, 1]), [0, 0]);
    }

    /// A decode step's claim queues its misses at the step rank, a keep or
    /// transient claim's at the prompt rank: the reader counts each kind
    /// alone, and a step's join publishes its picks pinned and counted as the
    /// step's fills.
    #[test]
    fn a_step_claims_misses_run_at_the_step_rank() {
        use std::sync::Mutex;
        use std::sync::atomic::Ordering::Relaxed;
        let arena = arena_of(8, 4);
        let books = Mutex::new(super::Books::default());
        let stats = super::Stats::default();
        let reader = super::Reader::start(2, "test-fill").unwrap();
        let counted = |r: &super::Reader| {
            let r = r.stats();
            ((r.step.jobs, r.step.units), (r.prompt.jobs, r.prompt.units))
        };
        let step = queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[0, 1],
            Claim::Step,
            |_, _| Ok(()),
        );
        let pin = step.finish().ok().expect("the step's fill ended");
        assert_eq!(
            counted(&reader),
            ((1, 2 * PARTS as u64), (0, 0)),
            "a step claim's misses ran as a step fill, three parts a pick"
        );
        assert_eq!(
            (stats.fills.load(Relaxed), stats.step_fills.load(Relaxed)),
            (2, 2),
            "and are the step's fills"
        );
        drop(pin);
        let keep = queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[2],
            Claim::Keep,
            |_, _| Ok(()),
        );
        drop(keep.finish().ok().expect("the keep claim's fill ended"));
        assert_eq!(
            counted(&reader),
            ((1, 2 * PARTS as u64), (1, PARTS as u64)),
            "a keep claim's as a prompt fill, three parts a pick"
        );
    }

    /// A victim's fill runs on the reader at the victim rank and ends
    /// unpinned: the slot is filled with no open read (a pin left would
    /// refuse the landing's release), the fill counts as a fill and not a
    /// union's, and a read that failed frees the slot and names the failure.
    #[test]
    fn a_victim_fill_publishes_unpinned() {
        use std::sync::Mutex;
        use std::sync::atomic::Ordering::Relaxed;
        let arena = arena_of(8, 4);
        let books = Mutex::new(super::Books::default());
        let stats = super::Stats::default();
        let reader = super::Reader::start(2, "test-fill").unwrap();
        let tier = super::TierParts {
            arena: &arena,
            books: &books,
            stats: &stats,
            reader: &reader,
        };
        let claim_of = |id: u32| {
            let mut b = books.lock().unwrap();
            super::claim_slots(0, &arena, &mut b, &[id], Claim::Step).unwrap()
        };
        super::fill_victims(tier, &claim_of(5).picks, |_, _| Ok(())).unwrap();
        assert_eq!(filled_ids(&arena), [5]);
        assert_eq!(open_reads(&arena, &[5]), [0], "no pin is left");
        assert_eq!(
            (stats.fills.load(Relaxed), stats.union_fills.load(Relaxed)),
            (1, 0),
            "a fill, not a union's"
        );
        let ran = reader.stats();
        assert_eq!(
            (
                (ran.victim.jobs, ran.victim.units),
                (ran.prompt.jobs, ran.prompt.units)
            ),
            ((1, PARTS as u64), (0, 0)),
            "the reader ran it as a victim fill, three parts a pick"
        );
        let failed = claim_of(6);
        let e = super::fill_victims(tier, &failed.picks, |_, _| {
            Err(crate::GpuError::shape(
                "NvTier::fill",
                "expert 6: a short read".to_string(),
            ))
        })
        .unwrap_err();
        assert!(e.to_string().contains("a short read"), "{e}");
        assert_eq!(filled_ids(&arena), [5], "the failed pick is not lent");
        assert_eq!(stats.fills.load(Relaxed), 1, "and not counted");
    }

    /// A read that failed reaches the join as the failure it raised, naming
    /// the pick's id, never a panic on the reader: the batch's slots are freed
    /// (the next claim takes them), the hits' pins close, and the seam names
    /// the layer and the id.
    #[test]
    fn a_fill_error_reaches_wait_by_name() {
        use model::moe::TierError;
        use std::sync::Mutex;
        let arena = arena_of(8, 4);
        let books = Mutex::new(super::Books::default());
        let stats = super::Stats::default();
        let reader = super::Reader::start(1, "test-fill").unwrap();
        ensure_as(&arena, &mut books.lock().unwrap(), &[0], Claim::Step);
        // Id 0 is a hit; 1 and 2 are the picks, and the second read fails.
        let inflight = queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[0, 1, 2],
            Claim::Keep,
            |pick, _| {
                if pick == 1 {
                    return Err(crate::GpuError::shape(
                        "NvTier::fill",
                        "expert 2: the run ends 0 B in, its part needs 1 B at 4096".to_string(),
                    ));
                }
                Ok(())
            },
        );
        let failed = inflight.finish().err().expect("the fill failed");
        assert_eq!(failed.id, 2, "the failing pick's id");
        match super::fill_error(3, failed) {
            TierError::Fill { layer, id, source } => {
                assert_eq!((layer, id), (3, 2));
                assert!(
                    source.to_string().contains("the run ends 0 B in"),
                    "{source}"
                );
            }
            other => panic!("not a fill failure: {other}"),
        }
        assert_eq!(
            filled_ids(&arena),
            [0],
            "the picks are not lent half-filled"
        );
        assert_eq!(open_reads(&arena, &[0]), [0], "the hit's pin closed");
        // No slot is left filling: a claim of four ids takes every slot.
        let again = super::claim_slots(
            0,
            &arena,
            &mut books.lock().unwrap(),
            &[1, 2, 3, 4],
            Claim::Step,
        )
        .expect("the slots are free again");
        assert_eq!(again.picks.len(), 4);
    }

    /// A ticket dropped without its wait gives its claim back once the fill
    /// ended: a fill that read is published and every pin closed; one that
    /// failed frees its slots.
    #[test]
    fn an_unwaited_ticket_gives_its_claim_back() {
        use std::sync::Mutex;
        let arena = arena_of(8, 6);
        let books = Mutex::new(super::Books::default());
        let stats = super::Stats::default();
        let reader = super::Reader::start(1, "test-fill").unwrap();
        ensure_as(&arena, &mut books.lock().unwrap(), &[0], Claim::Step);
        drop(queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[0, 1, 2],
            Claim::Keep,
            |_, _| Ok(()),
        ));
        assert_eq!(
            filled_ids(&arena),
            [0, 1, 2],
            "the read picks are published"
        );
        assert_eq!(open_reads(&arena, &[0, 1, 2]), [0, 0, 0], "and unpinned");
        drop(queued(
            &arena,
            &books,
            &stats,
            &reader,
            &[3, 4],
            Claim::Keep,
            |_, _| {
                Err(crate::GpuError::shape(
                    "NvTier::fill",
                    "a short read".to_string(),
                ))
            },
        ));
        assert_eq!(
            filled_ids(&arena),
            [0, 1, 2],
            "a failed fill's picks are freed"
        );
        assert_eq!(open_reads(&arena, &[3, 4]), [0, 0]);
        let free = super::claim_slots(0, &arena, &mut books.lock().unwrap(), &[3], Claim::Step);
        assert!(free.is_ok(), "no slot was left filling");
    }

    /// A pick over an evicted slot reads over its pages in place: the bytes
    /// it reads replace the evicted expert's, and the rest of each window
    /// keeps what it held (a returned page would read zero there) — evicted
    /// or not, the same.
    #[test]
    fn a_refill_of_an_evicted_slot_gives_no_pages_back() {
        use engram::direct::DirectFile;
        use std::sync::Arc;
        const PAGE: usize = DIRECT_ALIGN;
        let dir = std::env::temp_dir();
        let old = dir.join(format!("nvtier-old-{}", std::process::id()));
        let new = dir.join(format!("nvtier-new-{}", std::process::id()));
        std::fs::write(&old, vec![0xEE; PAGE]).unwrap();
        std::fs::write(&new, vec![0x11; 100]).unwrap();
        let (old, new) = (
            Arc::new(DirectFile::open_buffered(&old).unwrap()),
            Arc::new(DirectFile::open_buffered(&new).unwrap()),
        );
        let mut arena = arena_of(8, 2);
        arena.windows = [0, PAGE, 2 * PAGE];
        arena.slot_bytes = 3 * PAGE;
        arena.files = shards_of([new]);
        arena.parts = vec![[one_run(0, 0, 100, 0); PARTS]; 8];
        let mem = super::ArenaMem::new(AnonMap::new(2 * arena.slot_bytes).unwrap());
        for (slot, gone) in [(0, Some(3)), (1, None)] {
            let base = slot * arena.slot_bytes;
            for part in 0..PARTS {
                let got = mem.read_into(&old, 0, base + part * PAGE, PAGE).unwrap();
                assert_eq!(got, PAGE);
            }
            let pick = super::Pick { slot, id: 7, gone };
            super::read_pick(&mem, &super::read_of(&arena, &pick)).unwrap();
            for part in 0..PARTS {
                let at = base + part * PAGE;
                assert!(mem.slice(at, 100).iter().all(|&b| b == 0x11));
                assert!(
                    mem.slice(at + 100, PAGE - 100).iter().all(|&b| b == 0xEE),
                    "slot {slot} part {part}: the bytes beside the read were returned"
                );
            }
        }
    }

    /// A directory under the workspace's `target`, on the tree's disk (the
    /// system's temp directory can be a tmpfs, which may refuse an
    /// `O_DIRECT` open), removed on drop, a failed test's included.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target")
                .join(format!("nvtier-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The split a tier's dropper holds: one tensor in `scratch`, never dropped here.
    fn split_in(scratch: &Scratch) -> std::sync::Arc<gguf::Split> {
        use gguf::write::{Layout, TensorDecl, Writer};
        let path = scratch.0.join("model.gguf");
        let decl = TensorDecl {
            name: "x".into(),
            dims: vec![4],
            type_id: 0,
            nbytes: 16,
        };
        let layout = Layout::new(&[], vec![decl]).unwrap();
        let mut writer = Writer::new(std::fs::File::create(&path).unwrap(), layout).unwrap();
        writer.tensor("x", &[0; 16]).unwrap();
        writer.finish().unwrap();
        std::sync::Arc::new(gguf::Split::open(&path).unwrap())
    }

    /// A stack a checkpoint folder locates, its three matrices' codes and
    /// block scales at offsets off every page, in two shards
    /// ([`gguf::Place::Stack`]): matrix `e`'s codes in shard `e % 2`, its
    /// scales in the other.
    fn located_split(scratch: &Scratch) -> std::sync::Arc<gguf::Split> {
        use gguf::write::{Layout, TensorDecl};
        use gguf::{GENERAL_ARCHITECTURE, Place, StackLayout, Value};
        let files: Vec<_> = (0..2)
            .map(|shard| {
                let path = scratch.0.join(format!("folder-{shard}.bin"));
                let bytes: Vec<u8> = (0..8 * DIRECT_ALIGN)
                    .map(|i| ((i % 251) as u8).wrapping_add(97 * shard as u8 + 1))
                    .collect();
                std::fs::write(&path, bytes).unwrap();
                path
            })
            .collect();
        let kvs = [(
            GENERAL_ARCHITECTURE.to_string(),
            Value::String("test".into()),
        )];
        let decl = TensorDecl {
            name: "s.weight".into(),
            dims: vec![LOCATED_PER as u64 / 4, 3],
            type_id: 0,
            nbytes: 3 * LOCATED_PER as u64,
        };
        let layout = Layout::new(&kvs, vec![decl]).unwrap();
        let place = Place::Stack {
            codes_in: vec![0, 1, 0],
            codes_at: vec![
                37,
                4 * DIRECT_ALIGN as u64 + 1000,
                3 * DIRECT_ALIGN as u64 + 6,
            ],
            scales_in: vec![1, 0, 1],
            scales_at: vec![DIRECT_ALIGN as u64 - 3, 2 * DIRECT_ALIGN as u64 + 70, 7],
            codes_per: LOCATED_CODES as u64,
            scales_per: (LOCATED_PER - LOCATED_CODES) as u64,
            layout: StackLayout::Planes,
            scales: vec![1.0; 3],
        };
        let split = gguf::Split::located(files[0].clone(), &layout, &files, vec![place]).unwrap();
        std::sync::Arc::new(split)
    }

    /// A located matrix's bytes: its codes, then its block scales.
    const LOCATED_CODES: usize = 5000;
    const LOCATED_PER: usize = LOCATED_CODES + 704;

    /// A stack a checkpoint folder locates is read where the folder keeps
    /// it: each part of an expert is its matrix's two runs, each in its own
    /// shard ([`super::stack_runs`]), read into a window each and joined, so
    /// a pinned slot lends the matrix as `gguf::Split::matrix` reads it, its
    /// codes then its block scales, in a slot of the placement's part shape
    /// ([`window_bytes`]); and a drop names every run in its own shard. A
    /// stack the split does not locate stays one run of its file. FAIL-first:
    /// the arena's runs taken as one run of the file (`gguf::Split::run`)
    /// refuse the stack by name, and a fill without the join lends the codes
    /// and the bytes after them in the codes' window.
    #[test]
    fn a_located_stack_fills_each_matrix_from_its_two_shards() {
        use super::NvTier;
        use model::moe::TierSlots;
        use model::placement::workstation::ArenaGrant;
        let scratch = Scratch::new("located");
        let split = located_split(&scratch);
        let runs = super::stack_runs(&split, &routed(&split, "s.weight", None), 3)
            .expect("the folder's runs");
        let mut files = Vec::new();
        for r in runs
            .iter()
            .flat_map(|&(codes, scales)| std::iter::once(codes).chain(scales))
        {
            super::open_shard(&split, &mut files, r.shard, false, "s.weight").unwrap();
        }
        let window = window_bytes(LOCATED_CODES, LOCATED_PER - LOCATED_CODES);
        let mut arena = arena_of(3, 0);
        arena.windows = [0, window, 2 * window];
        arena.slot_bytes = PARTS * window;
        arena.files = files.into();
        arena.parts = runs
            .iter()
            .map(|&(codes, scales)| {
                let (_, _, need) = aligned_span(codes.at, codes.len as u64);
                [PartRange {
                    codes,
                    scales,
                    skew: need - codes.len,
                }; PARTS]
            })
            .collect();
        arena.part_of.set([0, 1, 2]).unwrap();
        let slots = [arena.slot_bytes];
        let grant = ArenaGrant::whole((2 * slots[0]) as u64);
        let tier = NvTier::carve(
            vec![Some(arena)],
            &slots,
            2,
            grant,
            0,
            &split,
            super::ArenaKind::Anon,
        )
        .expect("the tier's carve");
        for id in 0..3u32 {
            tier.ensure(0, &[id]).expect("the fill");
            let _pin = tier.pin(0, &[id]).expect("a filled id pins");
            let m = split.matrix("s.weight", id as usize).unwrap();
            let want = [m.codes, m.block_scales].concat();
            for part in 0..PARTS {
                assert_eq!(
                    TierSlots::slot(&tier, 0, id, part),
                    Some(&want[..]),
                    "expert {id} part {part}: the slot lends the matrix, codes then scales"
                );
            }
        }
        let shards: Vec<usize> = tier.drop_region(0).unwrap().iter().map(|r| r.0).collect();
        assert_eq!(
            shards,
            [0, 1, 0, 1].repeat(PARTS),
            "a row's codes then its scales, each run in its own shard"
        );
        let one = split_in(&scratch);
        let (shard, run) = one.run("x").unwrap();
        let want: Vec<_> = (0..4)
            .map(|e| (shard, run.start + 4 * e, 4, false))
            .collect();
        // The file's 16 bytes as four experts of one row each.
        let got: Vec<_> = super::stack_runs(&one, &routed(&one, "x", Some(vec![1, 4])), 4)
            .unwrap()
            .iter()
            .map(|(c, s)| (c.shard, c.at, c.len, s.is_some()))
            .collect();
        assert_eq!(got, want, "a stack of one file run cuts it evenly");
    }

    /// Routed stack `name` of `split` as a plan's tensor of layer 0, of `dims` (its header's when
    /// `None`).
    fn routed(
        split: &gguf::Split,
        name: &str,
        dims: Option<Vec<u64>>,
    ) -> model::placement::ModelTensor {
        let (shard, info) = split.find(name).expect("the stack");
        model::placement::ModelTensor {
            name: name.into(),
            shard,
            layer: Some(0),
            role: model::placement::Role::RoutedExperts,
            ty: info.ty,
            dims: dims.unwrap_or_else(|| info.dims.clone()),
            file_bytes: info.nbytes,
            gathered_rows: None,
        }
    }

    /// A family whose layers hold different expert counts pages each layer at its own: a layer
    /// of two experts among five of four, planned into a room a layer short of their host need
    /// (over the tier's floor of three layers), carves every layer's arena over its own count.
    /// FAIL-first: the arena took the model's count for every layer and refused the smaller
    /// layer's stacks ("2 experts, the model has 4").
    #[test]
    fn each_paged_layer_takes_its_own_expert_count() {
        use super::{ArenaKind, NvTier};
        use gguf::write::{Layout, TensorDecl, Writer};
        use model::placement::workstation::{self, ArenaGrant, HostNeed, HostRead};
        use model::placement::{
            KvBytes, ModelTensors, PlanLevers, expert_nvme_tier, plan_host_routed,
        };
        struct NoKv;
        impl KvBytes for NoKv {
            fn layer_bytes(&self, _: usize, _: u64) -> u64 {
                0
            }
        }
        // Q8_0 stacks of 32-wide rows, 64 rows an expert (2,176 B), an up and a down a layer.
        const ROWS: u64 = 64;
        let per = ROWS * 34;
        let counts = [4u64, 2, 4, 4, 4, 4];
        let scratch = Scratch::new("layer-experts");
        let path = scratch.0.join("model.gguf");
        let stacks: Vec<(usize, String, u64)> = counts
            .iter()
            .enumerate()
            .flat_map(|(l, &n)| ["up", "down"].map(|r| (l, format!("layer.{l}.{r}.weight"), n)))
            .collect();
        let decls = stacks
            .iter()
            .map(|(_, name, n)| TensorDecl {
                name: name.clone(),
                dims: vec![32, ROWS, *n],
                type_id: 8,
                nbytes: per * n,
            })
            .collect();
        let mut writer = Writer::new(
            std::fs::File::create(&path).unwrap(),
            Layout::new(&[], decls).unwrap(),
        )
        .unwrap();
        for (_, name, n) in &stacks {
            writer.tensor(name, &vec![1u8; (per * n) as usize]).unwrap();
        }
        writer.finish().unwrap();
        let split = std::sync::Arc::new(gguf::Split::open(&path).unwrap());
        let model = ModelTensors {
            tensors: stacks
                .iter()
                .map(|(l, name, _)| model::placement::ModelTensor {
                    layer: Some(*l),
                    ..routed(&split, name, None)
                })
                .collect(),
            layers: counts.len(),
            experts: counts[0],
            experts_used: 2,
            bf16_words: |_| false,
        };
        let machine = workstation::plan_on(workstation::A6000, counts.len());
        let mut plan =
            plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default()).expect("plans");
        // A room one layer short of the host need pages experts of every layer.
        let room = HostNeed::of(&plan, 0).bytes() - 2 * per * counts[0];
        let room = plan.plan_at(&(room, HostRead::Given));
        expert_nvme_tier(&mut plan, room, None).expect("the split");
        assert!(plan.host.nvme_expert_bytes > 0, "the room pages experts");
        let tier = NvTier::of_paged(
            &plan,
            &split,
            false,
            ArenaGrant::whole(64 << 20),
            ArenaKind::Anon,
            bloomery_levers::NvTierEvict::Tick,
        )
        .expect("the tier of two expert counts")
        .expect("a tier");
        for (l, &n) in counts.iter().enumerate() {
            let arena = tier.arena_of(l, "the test").expect("every layer pages");
            assert_eq!(
                u64::from(arena.n_expert),
                n,
                "layer {l}'s arena over its own count"
            );
        }
    }

    /// A relu² layer's arena holds two parts an expert, its up and its
    /// down, in a slot of their two windows: the leg's `[gate, up, down]`
    /// names its up in the gate slot again ([`NvTier::note_parts`]), so
    /// parts 0 and 1 lend the up's bytes and part 2 the down's, every fill
    /// reads the two runs alone, and a drop names the two rows. A gated
    /// layer's three rows refuse that leg by name (its gate row unnamed).
    /// FAIL-first: the arena's three parts an expert refused the leg ("names
    /// the plan's row twice").
    #[test]
    fn a_relu2_layer_pages_its_up_and_its_down() {
        use super::NvTier;
        use engram::direct::DirectFile;
        use model::moe::TierSlots;
        use model::placement::workstation::ArenaGrant;
        use std::sync::Arc;
        const LEN: usize = 3000;
        let scratch = Scratch::new("relu2");
        let split = split_in(&scratch);
        let path = scratch.0.join("rows");
        let bytes: Vec<u8> = (0..2 * 4 * LEN).map(|i| (i % 253) as u8 + 1).collect();
        std::fs::write(&path, &bytes).unwrap();
        let files = shards_of([Arc::new(DirectFile::open_buffered(&path).unwrap())]);
        let at = |row: usize, id: usize| (row * 4 + id) * LEN;
        let layer = |rows: usize| {
            let mut arena = arena_of(4, 0);
            arena.rows = rows;
            arena.names = ["up", "down", "gate"].map(String::from);
            arena.windows = [0, window_bytes(LEN, 0), 2 * window_bytes(LEN, 0)];
            arena.slot_bytes = rows * window_bytes(LEN, 0);
            arena.parts = (0..4)
                .map(|id| {
                    std::array::from_fn(|row| {
                        let (_, _, need) = aligned_span(at(row, id) as u64, LEN as u64);
                        one_run(0, at(row, id) as u64, LEN, need - LEN)
                    })
                })
                .collect();
            arena.files = Arc::clone(&files);
            arena
        };
        let carve = |arena: super::LayerArena| {
            let slots = [arena.slot_bytes];
            NvTier::carve(
                vec![Some(arena)],
                &slots,
                2,
                ArenaGrant::whole((2 * slots[0]) as u64),
                0,
                &split,
                super::ArenaKind::Anon,
            )
            .expect("the tier's carve")
        };
        let gated = carve(layer(3));
        let refused = gated
            .note_parts(0, &["up", "up", "down"])
            .expect_err("a gated layer's leg names its gate");
        assert!(
            refused
                .to_string()
                .contains("name none of the plan's row gate"),
            "{refused}"
        );
        let tier = carve(layer(2));
        tier.note_parts(0, &["up", "up", "down"])
            .expect("a relu² leg's up in the gate slot again");
        assert_eq!(tier.slot_set(), 2 * window_bytes(LEN, 0) as u64);
        for id in [1u32, 3] {
            tier.ensure(0, &[id]).expect("the fill");
            let _pin = tier.pin(0, &[id]).expect("a filled id pins");
            let row = |r: usize| Some(&bytes[at(r, id as usize)..at(r, id as usize) + LEN]);
            for (part, want) in [(0, row(0)), (1, row(0)), (2, row(1))] {
                assert_eq!(
                    TierSlots::slot(&tier, 0, id, part),
                    want,
                    "expert {id} part {part}: the up twice, then the down"
                );
            }
        }
        assert_eq!(tier.stats().fill_bytes, 2 * 2 * LEN as u64);
        assert_eq!(tier.drop_region(0).unwrap().len(), 2, "one drop a row");
    }

    /// `devmap` over the tier's own carve ([`NvTier::carve`]): two paged
    /// layers, so the second's base is past 0, each stack row its own file
    /// and range, the leg's parts resolved to other rows (`[2, 0, 1]`), and
    /// an expert in the slot before. The arena is what the lever's `pinned`
    /// asks of this device ([`arena_kind_of`]), allocated after a small
    /// allocation of the same kind, and write-combined on an integrated
    /// device alone. Each round fills expert 7's slot from files whose bytes
    /// differ from every other round's and row's at every offset, read
    /// `O_DIRECT` (`direct`) or through the page cache, pins it through the
    /// tier, and holds the card's read through the pin's address of each
    /// part, then the tier's host view ([`TierSlots::slot`]), to the file's
    /// bytes; the slot is given back between rounds. Then the lever's `anon`
    /// carves an arena whose pin gives no address, and `auto` pins on an
    /// integrated device alone.
    fn devmap_rounds(direct: bool) {
        use super::{ArenaKind, NvTier};
        use crate::graph::{MappedHost, cu};
        use crate::model::{PinnedSource, arena_kind_of};
        use bloomery_levers::NvTierMem;
        use engram::direct::DirectFile;
        use model::moe::TierSlots;
        use model::placement::workstation::ArenaGrant;
        use std::sync::Arc;
        use std::sync::atomic::Ordering::Relaxed;
        const PAGE: usize = DIRECT_ALIGN;
        const LAYER: usize = 1;
        // The plan's stack row each of the leg's parts reads.
        const PART_OF: [usize; PARTS] = [2, 0, 1];
        // Per stack row: the file range, inside one window at its skew.
        const ROWS: [(usize, usize); PARTS] =
            [(37, 100), (PAGE + 5, PAGE + 50), (2 * PAGE + 300, 200)];
        let bytes_of = |row: usize, round: usize| -> Vec<u8> {
            (0..3 * PAGE)
                .map(|i| ((i % 251) as u8).wrapping_add((83 * row + 29 * round + 1) as u8))
                .collect()
        };
        let (ctx, _stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let scratch = Scratch::new(if direct { "devmap-direct" } else { "devmap" });
        let paths: [_; PARTS] = std::array::from_fn(|row| scratch.0.join(format!("row{row}")));
        let split = split_in(&scratch);
        let layer = || {
            let mut arena = arena_of(8, 0);
            arena.windows = [0, PAGE, 3 * PAGE];
            arena.slot_bytes = 4 * PAGE;
            arena.parts = vec![
                std::array::from_fn(|row| one_run(
                    row,
                    ROWS[row].0 as u64,
                    ROWS[row].1,
                    ROWS[row].0 % PAGE
                ));
                8
            ];
            arena.part_of.set(PART_OF).unwrap();
            Some(arena)
        };
        let tier_of = |mem: ArenaKind<'_>| {
            let slots = [4 * PAGE; 2];
            let grant = ArenaGrant::whole((2 * slots.iter().sum::<usize>()) as u64);
            NvTier::carve(vec![layer(), layer()], &slots, 2, grant, 0, &split, mem)
                .expect("the tier's carve")
        };
        let fill = |tier: &mut NvTier, slot: usize, id: u32, round: usize| {
            for (row, path) in paths.iter().enumerate() {
                std::fs::write(path, bytes_of(row, round)).unwrap();
            }
            let arena = tier.by_layer[LAYER].as_mut().unwrap();
            arena.files = shards_of(paths.iter().map(|path| {
                Arc::new(if direct {
                    DirectFile::open(path).expect("a disk-backed file opens O_DIRECT")
                } else {
                    DirectFile::open_buffered(path).unwrap()
                })
            }));
            let pick = super::Pick {
                slot,
                id,
                gone: None,
            };
            super::read_pick(&tier.mem, &super::read_of(arena, &pick))
                .unwrap_or_else(|e| panic!("round {round} slot {slot}: the slot's fill: {e}"));
            arena.of_id[id as usize].store(slot as u32 + 1, Relaxed);
            arena.slots[slot].state.store(FILLED, Relaxed);
        };
        let file_part = |part: usize, round: usize| {
            let row = PART_OF[part];
            let (at, len) = ROWS[row];
            bytes_of(row, round)[at..at + len].to_vec()
        };
        // A small allocation of the same kind first, live for the test: a
        // driver that shares a chunk among small allocations puts the arena's
        // off a page boundary.
        let _before =
            MappedHost::read_in_place(&ctx, 1, "cuMemHostAlloc (before the test's arena)")
                .expect("a pinned byte");
        let source = PinnedSource(&ctx);
        let pinned = arena_kind_of(&source, NvTierMem::Pinned).unwrap();
        assert!(
            matches!(pinned, ArenaKind::Provided(_)),
            "the lever's pinned"
        );
        let mut tier = tier_of(pinned);
        let mut flags = 0u32;
        // SAFETY: the host address is the arena's first byte, inside the
        // allocation and live until `tier` drops, and `flags` is a local the
        // call writes.
        let rc = unsafe {
            cuda_core::sys::cuMemHostGetFlags(&mut flags, tier.mem.start() as *mut std::ffi::c_void)
        };
        cu(rc, "cuMemHostGetFlags").unwrap();
        let integrated = crate::raw_device_integrated(ctx.cu_device()).unwrap();
        assert_eq!(
            flags & cuda_core::sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0,
            integrated,
            "the arena is write-combined on an integrated device alone"
        );
        fill(&mut tier, 0, 3, 7);
        ctx.bind_to_thread().unwrap();
        for round in 0..3 {
            if round > 0 {
                let arena = tier.by_layer[LAYER].as_ref().unwrap();
                tier.mem
                    .give_back(arena.slot_off(1), arena.slot_bytes)
                    .expect("the give-back of a pinned slot");
            }
            fill(&mut tier, 1, 7, round);
            let pin = tier.pin(LAYER, &[7]).unwrap();
            for part in 0..PARTS {
                let want = file_part(part, round);
                let dev = pin.device_slot(7, part).expect("a pinned slot's address");
                let mut got = vec![0u8; want.len()];
                // SAFETY: the context is current (bound above); `dev` addresses
                // `got.len()` bytes of the allocation, and `got` holds them.
                let rc = unsafe {
                    cuda_core::sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dev, got.len())
                };
                cu(rc, "cuMemcpyDtoH_v2").unwrap();
                assert_eq!(
                    got, want,
                    "round {round} part {part}: the card's read of the slot"
                );
                assert_eq!(
                    tier.slot(LAYER, 7, part),
                    Some(&want[..]),
                    "round {round} part {part}: the host's view"
                );
            }
            assert_eq!(
                pin.device_slot(3, 0),
                None,
                "an id the pin does not hold has no address"
            );
        }
        let anon = arena_kind_of(&source, NvTierMem::Anon).unwrap();
        assert!(matches!(anon, ArenaKind::Anon), "the lever's anon");
        let mut tier = tier_of(anon);
        fill(&mut tier, 1, 7, 0);
        let pin = tier.pin(LAYER, &[7]).unwrap();
        for part in 0..PARTS {
            assert_eq!(
                pin.device_slot(7, part),
                None,
                "part {part}: an anonymous arena's slot has no card address"
            );
            assert_eq!(
                tier.slot(LAYER, 7, part),
                Some(&file_part(part, 0)[..]),
                "part {part}: the anonymous arena's host view"
            );
        }
        assert_eq!(
            matches!(
                arena_kind_of(&source, NvTierMem::Auto).unwrap(),
                ArenaKind::Provided(_)
            ),
            integrated,
            "auto pins on an integrated device alone"
        );
    }

    /// `devmap`: the card's address of a pinned arena's filled slot reads
    /// the bytes the slot lends, after the first fill and after each release
    /// and refill through the page cache: the release keeps the pages, so
    /// the refill lands in the pages the card reads. On an integrated device
    /// and on a card across PCIe alike.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_devmap_a_pinned_slot_reads_on_the_card_as_on_the_host_across_a_release() {
        devmap_rounds(false);
    }

    /// `devmap` with the tier's default reads: `O_DIRECT` fills land in the
    /// pinned slot, and the card reads them as the host does.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_devmap_a_direct_fill_lands_in_a_pinned_slot() {
        devmap_rounds(true);
    }

    /// The card hold tests' layer: eight experts of one page a part, each
    /// stack row in a file of its own, held by a tier of two slots.
    const CARD_PAGE: usize = DIRECT_ALIGN;
    const CARD_LEN: usize = 64;

    /// Expert `id`'s part of stack row `row`: one page an expert, a
    /// word-aligned skew a row; expert 7's skew is off a word.
    fn card_at(row: usize, id: usize) -> usize {
        id * CARD_PAGE + if id == 7 { 37 } else { 4 * (row + 1) }
    }

    fn card_bytes(row: usize) -> Vec<u8> {
        (0..8 * CARD_PAGE)
            .map(|i| ((i % 251) as u8).wrapping_add((83 * row + 1) as u8))
            .collect()
    }

    /// An anonymous tier of two slots whose layer 1 is the card hold tests'
    /// layer.
    fn card_tier(scratch: &Scratch) -> std::sync::Arc<super::NvTier> {
        eight_tier(scratch, 2, NvTierEvict::Tick)
    }

    /// An anonymous tier of `count` slots whose layer 1 is the card hold
    /// tests' layer, its victims taken as `evict` says.
    fn eight_tier(
        scratch: &Scratch,
        count: usize,
        evict: NvTierEvict,
    ) -> std::sync::Arc<super::NvTier> {
        eight_tier_over(scratch, count, evict, super::ArenaKind::Anon)
    }

    /// [`eight_tier`] over the arena memory `mem` says.
    fn eight_tier_over(
        scratch: &Scratch,
        count: usize,
        evict: NvTierEvict,
        mem: super::ArenaKind<'_>,
    ) -> std::sync::Arc<super::NvTier> {
        use super::NvTier;
        use engram::direct::DirectFile;
        use model::placement::workstation::ArenaGrant;
        use std::sync::Arc;
        let split = split_in(scratch);
        let path = scratch.0.join("rows");
        std::fs::write(&path, (0..PARTS).flat_map(card_bytes).collect::<Vec<u8>>()).unwrap();
        let files = shards_of([Arc::new(DirectFile::open_buffered(&path).unwrap())]);
        let layer = || {
            let mut arena = arena_of(8, 0);
            arena.windows = [0, CARD_PAGE, 2 * CARD_PAGE];
            arena.slot_bytes = 3 * CARD_PAGE;
            arena.parts = (0..8)
                .map(|id| {
                    std::array::from_fn(|row| {
                        let at = row * card_bytes(row).len() + card_at(row, id);
                        one_run(0, at as u64, CARD_LEN, card_at(row, id) % CARD_PAGE)
                    })
                })
                .collect();
            arena.files = Arc::clone(&files);
            arena.part_of.set([0, 1, 2]).unwrap();
            Some(arena)
        };
        let slots = [3 * CARD_PAGE];
        let grant = ArenaGrant::whole((count * slots[0]) as u64);
        let mut tier = NvTier::carve(vec![None, layer()], &slots, count, grant, 0, &split, mem)
            .expect("the tier's carve");
        if evict == NvTierEvict::Heat {
            tier.count_routes();
        }
        Arc::new(tier)
    }

    /// A card hold ([`NvTier::hold_for_card`]) fills and pins its ids for as
    /// long as it lives: no claim takes their slots while it does, its drop
    /// gives them back to the LRU, and the step's fills and the card's ids
    /// are counted. A hold of a miss and a hit names them in the order asked.
    /// Each part's word offset is its slot's first byte in the
    /// arena over four, a part that does not start on a word is refused by
    /// name, and the layer then has parts off a word. The arena is anonymous,
    /// so the card has no address of it. The card leg's state reads
    /// unattached until the load notes it, once.
    #[test]
    fn a_card_hold_pins_its_ids_until_it_drops() {
        use super::LegOff;
        use model::moe::TierSlots;
        const LAYER: usize = 1;
        let scratch = Scratch::new("card-hold");
        let tier = card_tier(&scratch);
        let hold = tier.hold_for_card(LAYER, &[1, 2]).unwrap();
        assert_eq!(hold.ids(), [1, 2]);
        for id in [1u32, 2] {
            let offs = hold.offsets(id).unwrap();
            for (p, off) in offs.iter().enumerate() {
                let part = tier.slot(LAYER, id, p).expect("a held id's slot");
                assert_eq!(
                    *off * 4,
                    (part.as_ptr() as usize - tier.mem.start()) as u64,
                    "expert {id} part {p}: the word offset of its slot's bytes"
                );
                let a = card_at(p, id as usize);
                assert_eq!(
                    part,
                    &card_bytes(p)[a..a + CARD_LEN],
                    "expert {id} part {p}: the file's bytes"
                );
            }
        }
        let e = tier
            .ensure_pinned(LAYER, &[3], Claim::Step)
            .map(|_| ())
            .expect_err("both slots are held")
            .to_string();
        assert!(e.contains("pinned by a lane's read or a batch pin"), "{e}");
        let s = tier.stats();
        assert_eq!((s.card_ids, s.step_fills), (2, 2));
        drop(hold);
        let buffers = || {
            let picks = tier.step.picks.lock().unwrap();
            let evicted = tier.step.evicted.lock().unwrap();
            let reads = tier.step.buf.reads.lock().unwrap();
            (
                (picks.as_ptr() as usize, picks.capacity()),
                (evicted.as_ptr() as usize, evicted.capacity()),
                (reads.as_ptr() as usize, reads.capacity()),
            )
        };
        let warmed = buffers();
        drop(
            tier.ensure_pinned(LAYER, &[3], Claim::Step)
                .expect("the dropped hold's slots are the LRU's again"),
        );
        let mixed = tier.hold_for_card(LAYER, &[5, 3]).unwrap();
        assert_eq!(
            mixed.ids(),
            [5, 3],
            "the asked order, the hit after the miss"
        );
        {
            let reads = tier.step.buf.reads.lock().unwrap();
            assert_eq!(reads.len(), 1, "the resident fill reads only the miss");
            assert_eq!(reads[0].id, 5, "the card hold uses the resident step fill");
        }
        assert_eq!(buffers(), warmed, "a warmed eviction reuses its buffers");
        drop(mixed);
        let odd = tier.hold_for_card(LAYER, &[7]).unwrap();
        let e = odd.offsets(7).unwrap_err().to_string();
        assert!(e.contains("off a word"), "{e}");
        drop(odd);
        assert!(
            !tier.parts_on_words(LAYER),
            "expert 7's parts start off a word"
        );
        assert!(!tier.parts_on_words(0), "a layer the arena does not cover");
        assert!(arena_of(8, 4).on_words(), "every part on a word");
        assert_eq!(tier.card_leg(), Some(LegOff::Unattached));
        tier.note_card_leg(Some(LegOff::Anon)).unwrap();
        assert_eq!(tier.card_leg(), Some(LegOff::Anon));
        let e = tier.note_card_leg(None).unwrap_err().to_string();
        assert!(e.contains("noted twice"), "{e}");
        assert_eq!(
            tier.card_arena(),
            None,
            "an anonymous arena has no card address"
        );
    }

    /// Pages over an anonymous map, `skew` bytes in, of `len` bytes, whose
    /// first byte the card would address as `dev`: a page source with no card
    /// behind it.
    struct HostPages {
        map: AnonMap,
        skew: usize,
        len: usize,
        dev: Option<u64>,
    }

    impl HostPages {
        fn new(skew: usize, len: usize, dev: Option<u64>) -> HostPages {
            let map = AnonMap::new(len + skew + DIRECT_ALIGN).expect("an anonymous map");
            HostPages {
                map,
                skew,
                len,
                dev,
            }
        }
    }

    impl super::ArenaPages for HostPages {
        fn base(&self) -> *mut u8 {
            self.map.as_ptr().cast_mut().wrapping_add(self.skew)
        }

        fn bytes(&self) -> usize {
            self.len
        }

        fn device_base(&self) -> Option<u64> {
            self.dev
        }
    }

    /// A [`HostPages`] source that records the first byte it gave.
    struct HostSource {
        skew: usize,
        dev: Option<u64>,
        gave: std::sync::atomic::AtomicUsize,
    }

    impl super::PageSource for HostSource {
        fn alloc(&self, len: usize) -> Result<Box<dyn super::ArenaPages>, crate::GpuError> {
            let pages = HostPages::new(self.skew, len, self.dev);
            let base = super::ArenaPages::base(&pages) as usize;
            self.gave.store(base, Ordering::Relaxed);
            Ok(Box::new(pages))
        }
    }

    /// A provided arena is its source's pages and nothing else: the fills
    /// land from the first byte the source names (a page in, so not the
    /// mapping's own), the card's address of a slot is the source's device
    /// base plus the slot's offset, and no give-back advises the pages (an
    /// anonymous arena's would read zeroes after one).
    #[test]
    fn a_provided_arena_is_its_sources_pages() {
        use super::ArenaKind;
        use model::moe::TierSlots;
        const LAYER: usize = 1;
        const DEV: u64 = 0xD000_0000;
        let scratch = Scratch::new("provided");
        let source = HostSource {
            skew: CARD_PAGE,
            dev: Some(DEV),
            gave: std::sync::atomic::AtomicUsize::default(),
        };
        let tier = eight_tier_over(&scratch, 2, NvTierEvict::Tick, ArenaKind::Provided(&source));
        assert_eq!(
            tier.mem.start(),
            source.gave.load(Ordering::Relaxed),
            "the arena starts at the first byte the source named"
        );
        assert_eq!(tier.mem.len(), 2 * 3 * CARD_PAGE);
        assert_eq!(
            tier.card_arena(),
            Some((DEV, 2 * 3 * CARD_PAGE / 4)),
            "the card's address of the arena is the source's device base"
        );
        let hold = tier.hold_for_card(LAYER, &[1, 2]).unwrap();
        let pin = tier.pin(LAYER, &[1, 2]).unwrap();
        let check = |what: &str| {
            for id in [1u32, 2] {
                for p in 0..PARTS {
                    let part = tier.slot(LAYER, id, p).expect("a held id's slot");
                    let a = card_at(p, id as usize);
                    assert_eq!(
                        part,
                        &card_bytes(p)[a..a + CARD_LEN],
                        "{what}: expert {id} part {p}: the file's bytes"
                    );
                    let at = (part.as_ptr() as usize - tier.mem.start()) as u64;
                    assert_eq!(
                        pin.device_slot(id, p),
                        Some(DEV + at),
                        "{what}: expert {id} part {p}: the card's address"
                    );
                }
            }
        };
        check("filled");
        tier.mem
            .give_back(0, tier.mem.len())
            .expect("the give-back of provided pages");
        check("after a give-back");
        drop((pin, hold));
    }

    /// A source with no device address gives the card none, and the host
    /// reads its slots the same.
    #[test]
    fn a_provided_arena_without_a_device_base_has_no_card_address() {
        use super::ArenaKind;
        use model::moe::TierSlots;
        const LAYER: usize = 1;
        let scratch = Scratch::new("provided-host");
        let source = HostSource {
            skew: 0,
            dev: None,
            gave: std::sync::atomic::AtomicUsize::default(),
        };
        let tier = eight_tier_over(&scratch, 2, NvTierEvict::Tick, ArenaKind::Provided(&source));
        assert_eq!(tier.card_arena(), None);
        let hold = tier.hold_for_card(LAYER, &[1]).unwrap();
        let pin = tier.pin(LAYER, &[1]).unwrap();
        for p in 0..PARTS {
            assert_eq!(pin.device_slot(1, p), None, "part {p}: no card address");
            let a = card_at(p, 1);
            assert_eq!(
                tier.slot(LAYER, 1, p),
                Some(&card_bytes(p)[a..a + CARD_LEN]),
                "part {p}: the host's view"
            );
        }
        drop((pin, hold));
    }

    /// A source that gives other than the bytes the carve asked for breaks
    /// its contract: refused by name, never an arena of another size.
    #[test]
    #[should_panic(expected = "the page source gave")]
    fn a_provided_arena_refuses_pages_of_another_length() {
        let pages = HostPages::new(0, 4 * DIRECT_ALIGN - 1, None);
        super::ArenaMem::provided(Box::new(pages), 4 * DIRECT_ALIGN);
    }

    /// A source whose first byte is off a [`DIRECT_ALIGN`] boundary breaks
    /// its contract: the direct fills' windows start on it, so it is refused
    /// by name.
    #[test]
    #[should_panic(expected = "is not 4096-aligned")]
    fn a_provided_arena_refuses_a_first_byte_off_the_direct_alignment() {
        let pages = HostPages::new(1, 4 * DIRECT_ALIGN, None);
        super::ArenaMem::provided(Box::new(pages), 4 * DIRECT_ALIGN);
    }

    /// A heat arena's victim order beside the gate's (`gate_nvtier`'s
    /// `evict` clause, which checks the tick and heat orders): a call's own
    /// hit is never its victim, however cold, and an evicted expert keeps
    /// its count.
    #[test]
    fn a_heat_arena_evicts_the_coldest_count() {
        const LAYER: usize = 1;
        let scratch = Scratch::new("heat-evict");
        let tier = eight_tier(&scratch, 4, NvTierEvict::Heat);
        // Expert 0 three times, then 1, 2 and 3 once each: the four slots full.
        for id in [0, 0, 0, 1, 2, 3] {
            tier.ensure(LAYER, &[id]).unwrap();
        }
        // Expert 1 is the coldest; the step that names it misses 7, which
        // takes 2, the next coldest.
        tier.ensure(LAYER, &[1, 7]).unwrap();
        let filled: Vec<u32> = (0..8).filter(|&id| tier.slot_filled(LAYER, id)).collect();
        assert_eq!(filled, [0, 1, 3, 7], "the call's own hit stays");
        let count = |id| tier.route_count(LAYER, id).expect("a heat tier counts");
        // Expert 2 went two steps after its one column: 2^(-2/H) of it.
        assert!(
            (count(2) - (-2.0 / super::HEAT_HALF_LIFE).exp2()).abs() < 1e-12,
            "{}",
            count(2)
        );
        assert!(count(0) > count(1) && count(1) > count(3) && count(3) > count(2));
    }

    /// The routing counts: a step's claim counts each id it names once and
    /// moves the layer's clock one column, so a count halves every
    /// `HEAT_HALF_LIFE` columns; a prompt's union call counts each id by its columns and moves
    /// the clock by the call's; a residency victim's fill counts one column
    /// and moves no clock; a refused claim counts nothing; a tick arena
    /// counts nothing. Past the rescale every count and rank holds.
    #[test]
    fn a_step_union_counts_each_experts_columns_on_both_prefetch_paths() {
        use model::moe::TierSlots;
        const LAYER: usize = 1;
        let scratch = Scratch::new("heat-step-union");
        let decay = (-4.0 / super::HEAT_HALF_LIFE).exp2();
        for allocating in [false, true] {
            let mut tier = eight_tier(&scratch, 4, NvTierEvict::Heat);
            std::sync::Arc::get_mut(&mut tier).unwrap().by_layer[LAYER]
                .as_mut()
                .unwrap()
                .host = 0..1;
            for id in [1, 2].into_iter().chain([3; 10]).chain([4; 10]) {
                tier.ensure(LAYER, &[id]).unwrap();
            }
            let before: Vec<_> = (0..8)
                .map(|id| tier.route_count(LAYER, id).unwrap())
                .collect();
            let clock = || tier.books.lock().unwrap().counts.as_ref().unwrap()[LAYER].since;
            let start = clock();
            if allocating {
                assert!(tier.step.try_begin());
            }
            // Four columns route [0, 1, 2], [1], [1], [1].
            let pin = tier
                .prefetch_pinned(
                    LAYER,
                    &[0, 1, 2],
                    Claim::Step,
                    4,
                    Some((&[0, 1, 5, 6], &[0, 0, 1, 2, 3, 0])),
                )
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(clock() - start, 4, "allocating={allocating}: column clock");
            // PIN(2026-10-23): each selection decays from its column,
            // rather than every selection entering at the call's end.
            let four: f64 = (0..4)
                .map(|col| ((col as f64 - 3.0) / super::HEAT_HALF_LIFE).exp2())
                .sum();
            let first = (-3.0 / super::HEAT_HALF_LIFE).exp2();
            for (id, added) in [(0, 0.0), (1, four), (2, first), (3, 0.0), (4, 0.0)] {
                let count = tier.route_count(LAYER, id).unwrap();
                let want = before[id as usize] * decay + added;
                assert!(
                    (count - want).abs() <= 1e-12 * want.abs(),
                    "allocating={allocating}: id {id}: {count} != {want}"
                );
            }
            assert_eq!(pin.ids(), [1, 2]);
            drop(pin);
            // The next miss takes 2, not the older but more frequent 1.
            let pin = tier
                .prefetch_pinned(
                    LAYER,
                    &[0, 5],
                    Claim::Step,
                    4,
                    Some((&[0, 2, 6], &[0, 1, 0, 1, 2, 3])),
                )
                .unwrap()
                .wait()
                .unwrap();
            assert!(
                !tier.slot_filled(LAYER, 2),
                "allocating={allocating}: cold victim"
            );
            assert!(tier.slot_filled(LAYER, 1) && tier.slot_filled(LAYER, 5));
            assert_eq!(clock() - start, 8, "allocating={allocating}: miss clock");
            assert_eq!(tier.route_count(LAYER, 5), Some(four));
            assert_eq!(tier.route_count(LAYER, 0), Some(0.0));
            drop(pin);
            if allocating {
                tier.step.release();
            }
        }
    }

    #[test]
    fn the_routing_counts_follow_the_columns() {
        use model::moe::TierSlots;
        const LAYER: usize = 1;
        // Relative error, so the counts below one after a rescale are checked too.
        let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
        let scratch = Scratch::new("heat-counts");
        let tier = eight_tier(&scratch, 4, NvTierEvict::Heat);
        let count = |id| tier.route_count(LAYER, id).expect("a heat tier counts");
        tier.ensure(LAYER, &[1, 1, 2]).unwrap();
        assert_eq!(
            (count(1), count(2), count(3)),
            (1.0, 1.0, 0.0),
            "once an id"
        );
        // A union of one half-life's columns lists 1 in a quarter of them and
        // 3 in the rest.
        let h = super::HEAT_HALF_LIFE as usize;
        tier.note_union(
            LAYER,
            &[1, 3],
            &[0, h / 4, h],
            &(0..h).collect::<Vec<_>>(),
            h,
            true,
        );
        // PIN(2026-10-23): the count weights each original column,
        // replacing the call-end lump of h/4 and h-h/4 selections.
        let weight = |col: usize| ((col as f64 + 1.0 - h as f64) / super::HEAT_HALF_LIFE).exp2();
        let quarter: f64 = (0..h / 4).map(weight).sum();
        let rest: f64 = (h / 4..h).map(weight).sum();
        assert!(near(count(1), quarter + 0.5) && near(count(2), 0.5) && near(count(3), rest));
        tier.ensure_victim(LAYER, &[4]).unwrap();
        assert!(
            near(count(4), 1.0) && near(count(2), 0.5),
            "the clock stays"
        );
        tier.ensure(LAYER, &[1, 8]).unwrap_err();
        assert!(
            near(count(1), quarter + 0.5),
            "a refused claim counts nothing"
        );
        assert_eq!(
            eight_tier(&scratch, 4, NvTierEvict::Tick).route_count(LAYER, 1),
            None
        );
        let mut c = super::LayerCounts::new(2);
        c.add(0, 1.0);
        c.advance(h as u64);
        c.add(1, 1.0);
        c.advance(super::HEAT_RESCALE as u64 * h as u64);
        assert_eq!(c.since, 0, "rescaled");
        assert!(near(c.count(0), (-65.0f64).exp2()) && near(c.count(1), (-64.0f64).exp2()));
        assert!(c.score(0) < c.score(1));
    }

    /// How far the test has let the held reads go.
    #[derive(Clone, Copy, PartialEq, PartialOrd)]
    enum Phase {
        Held,
        /// Every held read but the first unit's may end.
        Open,
        /// The first unit's read may end too.
        Ended,
    }

    /// What the held reads wait on: the phase, and whether a read waited past
    /// its bound.
    struct Latch {
        phase: std::sync::Mutex<Phase>,
        moved: std::sync::Condvar,
        expired: std::sync::atomic::AtomicBool,
    }

    /// Every reader thread of a tier held in a read the test lets go, so a
    /// step's misses queue behind them and none starts until [`Gate::open`].
    /// The reads are one job whose books the test closes ([`Gate::end`]): the
    /// tier's stop counts a job still open as a ticket that outlived the tier.
    struct Gate {
        latch: std::sync::Arc<Latch>,
        job: std::sync::Arc<super::Job>,
    }

    impl Gate {
        /// Hold every reader thread of `tier` in a read, and return once each
        /// has one. A read the gate holds past five seconds ends and marks the
        /// gate expired, so a caller that waited on its fill fails its test
        /// rather than hanging it.
        fn close(tier: &super::NvTier) -> Gate {
            use super::{Job, Kind, NVTIER_FILL_THREADS};
            let latch = std::sync::Arc::new(Latch {
                phase: std::sync::Mutex::new(Phase::Held),
                moved: std::sync::Condvar::new(),
                expired: std::sync::atomic::AtomicBool::new(false),
            });
            let (taken_tx, taken) = std::sync::mpsc::channel();
            let held = std::sync::Arc::clone(&latch);
            let job = Job::new(Kind::Victim, NVTIER_FILL_THREADS, move |unit| {
                taken_tx.send(()).unwrap();
                let until = if unit == 0 { Phase::Ended } else { Phase::Open };
                let phase = held.phase.lock().unwrap();
                let (_phase, wait) = held
                    .moved
                    .wait_timeout_while(phase, std::time::Duration::from_secs(5), |p| *p < until)
                    .unwrap();
                if wait.timed_out() {
                    held.expired
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }
                Ok(())
            });
            let job = tier.reader.submit(job).expect("the gate's job");
            for _ in 0..NVTIER_FILL_THREADS {
                taken
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .expect("every reader thread took a unit");
            }
            Gate { latch, job }
        }

        fn move_to(&self, to: Phase) {
            *self.latch.phase.lock().unwrap() = to;
            self.latch.moved.notify_all();
        }

        /// Let go of every held read but one, which stays in its read until
        /// [`Gate::end`]: the fill runs on the readers let go and ends while
        /// that read is still in flight, as it can when the others wake late.
        fn open(&self) {
            self.move_to(Phase::Open);
        }

        /// Let go of the last held read and wait for the gate's job to close:
        /// no read of the gate is left when the test's tier stops.
        fn end(&self, tier: &super::NvTier) {
            self.move_to(Phase::Ended);
            tier.reader.wait(&self.job).expect("the gate's reads ended");
        }

        fn expired(&self) -> bool {
            self.latch.expired.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    /// A card ticket gives the caller its thread back while the misses'
    /// reads are still queued behind the held readers, and its slots read
    /// nothing until [`CardTicket::finish`] joins the fill: the work the
    /// caller runs in between is counted as host compute under the fill, the
    /// ids count as the card's only at the finish, and the hold the finish
    /// returns names the ids in the order asked, hands the card the offsets
    /// of exactly those, and refuses an id it does not pin. The finish joins
    /// the fill while one held read is still in flight, so the fill needs no
    /// reader but the ones the test let go.
    #[test]
    fn a_card_ticket_hands_the_caller_back_before_its_fill() {
        const LAYER: usize = 1;
        let scratch = Scratch::new("card-ticket");
        let tier = card_tier(&scratch);
        let gate = Gate::close(&tier);
        let ids = [2u32, 1];
        let ticket = tier.issue_card_hold(LAYER, &ids).expect("the claim");
        assert!(
            !gate.expired(),
            "the claim waited for its fill: the readers were still held"
        );
        assert!(!ticket.ticket.is_ready(), "two misses: a fill in flight");
        assert!(
            !tier.slot_filled(LAYER, 1) && !tier.slot_filled(LAYER, 2),
            "no slot is filled while the readers are held"
        );
        let worked = ticket.under_fill(|| {
            std::thread::sleep(std::time::Duration::from_millis(2));
            7
        });
        assert_eq!(worked, 7, "the work's result passes through");
        let s = tier.stats();
        assert!(
            s.step_hits_under_read_ns >= 1_000_000,
            "the work ran under the fill: {} ns",
            s.step_hits_under_read_ns
        );
        assert_eq!(s.card_ids, 0, "no id is the card's before the hold");
        gate.open();
        let hold = ticket.finish().expect("the fill ended");
        assert!(!gate.expired(), "the readers were released by the test");
        assert_eq!(hold.ids(), [2, 1], "the order asked");
        assert!(tier.slot_filled(LAYER, 1) && tier.slot_filled(LAYER, 2));
        assert!(hold.offsets(2).is_ok() && hold.offsets(1).is_ok());
        let e = hold.offsets(3).unwrap_err().to_string();
        assert!(e.contains("the hold pins"), "{e}");
        assert_eq!(tier.stats().card_ids, 2, "the finish counts the card's ids");
        gate.end(&tier);
        assert!(!gate.expired(), "the last read was let go by the test");
    }

    /// A card ticket dropped without a finish (the host's work between the
    /// issue and the finish failed) joins its fill and gives its slots back:
    /// the layer's two slots serve a claim of two other ids, the step slot
    /// takes the next ticket, and only the finished hold's ids are the card's.
    #[test]
    fn a_dropped_card_ticket_gives_its_slots_back() {
        const LAYER: usize = 1;
        let scratch = Scratch::new("card-ticket-drop");
        let tier = card_tier(&scratch);
        let ticket = tier.issue_card_hold(LAYER, &[1, 2]).expect("the claim");
        drop(ticket);
        drop(
            tier.ensure_pinned(LAYER, &[3, 4], Claim::Step)
                .expect("the dropped ticket's slots are the LRU's again"),
        );
        let next = tier
            .issue_card_hold(LAYER, &[5])
            .expect("the step slot is free again");
        drop(next.finish().expect("a hold after the drop"));
        assert_eq!(tier.stats().card_ids, 1, "only the finished hold counted");
    }

    /// The arena entries over a pinned tier's slot that each round refills
    /// in place with another expert, through the tier's own fill (`O_DIRECT`
    /// and through the page cache): right after each refill, the Q4_K
    /// gate·up and Q5_K down arena entries, eager and from a graph captured
    /// in the first round, read the slot through its pin's addresses and
    /// write that round's expert's rows, the stack `_sel` launches' over the
    /// same bytes bit for bit and never the round before's. The parts are a
    /// few hundred KiB, so a line of the expert before can still sit in the
    /// card's cache when the refill lands; nothing else reads the arena in
    /// between.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_refilled_pinned_slot_reads_as_its_new_expert_in_the_arena_entries() {
        use super::NvTier;
        use crate::Gpu;
        use crate::card_leg::tests::{stack_bytes, words_of};
        use crate::graph::Graph;
        use crate::kquant::{
            Act, GateUpAct, GateUpAt, PART_DOWN, PART_GATE, PART_UP, SelDown, SelDownAt,
        };
        use crate::q4k_sel::QuantSel;
        use crate::quant_sel::QuantSelKernels;
        use crate::tensor::{DeviceTensor, Q8Act, window};
        use cuda_core::DeviceBuffer;
        use engram::direct::DirectFile;
        use gguf::GgmlType;
        use model::placement::workstation::ArenaGrant;
        use std::sync::Arc;
        use std::sync::atomic::Ordering::Relaxed;
        const PAGE: usize = DIRECT_ALIGN;
        const LAYER: usize = 1;
        const ID: u32 = 7;
        const ROUNDS: usize = 4;
        // The model width and the routed width: two and one super-blocks.
        const N: usize = 512;
        const FF: usize = 256;
        // Per part (gate, up, down): its type, rows and k.
        const ROLES: [(GgmlType, usize, usize); PARTS] = [
            (GgmlType::Q4_K, FF, N),
            (GgmlType::Q4_K, FF, N),
            (GgmlType::Q5_K, N, FF),
        ];
        // Each part's first byte in its file: on a word, off the page.
        const AT: [usize; PARTS] = [1248, PAGE + 36, 2 * PAGE + 4000];
        let expert = |round: usize, p: usize| {
            stack_bytes(
                ROLES[p].0,
                ROLES[p].1,
                ROLES[p].2,
                (round * PARTS + p) as u64 + 0x5eed,
            )
        };
        let len: [usize; PARTS] = std::array::from_fn(|p| expert(0, p).len());
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let kq = QuantSelKernels::load(ctx, gpu.fault_word()).expect("the kquant module");
        let sink = gpu.layer_sink(0).expect("a sink");
        let rule = Act::SwigluClamp { limit: 0.25 };
        let normed: Vec<f32> = (0..N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &normed).expect("the normed row");
        let mut act_x = Q8Act::with_tier_cols(stream, 1, N).expect("act_x");
        gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, 1, 0)
            .expect("q8_1");
        let one = DeviceBuffer::from_host(stream, &[0u32]).expect("the place");

        // The reference: the rounds' experts as stacks, round r's at id r.
        let stacks: Vec<DeviceTensor<u32>> = (0..PARTS)
            .map(|p| {
                let rows = ROUNDS * ROLES[p].1;
                let bytes: Vec<u8> = (0..ROUNDS).flat_map(|r| expert(r, p)).collect();
                let words = words_of(&bytes, rows);
                let n = words.len();
                DeviceTensor::upload(stream, &words, rows, n / rows).expect("a stack")
            })
            .collect();
        let want: Vec<(Vec<f32>, Vec<f32>)> = (0..ROUNDS)
            .map(|r| {
                let sel = DeviceBuffer::from_host(stream, &[r as u32]).expect("the id");
                let mut h = DeviceBuffer::<f32>::zeroed(stream, FF).expect("h");
                let mut act_h = Q8Act::with_slots(stream, 1, FF).expect("act_h");
                let mut y = DeviceBuffer::<f32>::zeroed(stream, N).expect("y");
                let a = GateUpAct {
                    wg: &stacks[PART_GATE],
                    wu: &stacks[PART_UP],
                    act: &act_x,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: FF,
                    slots_per_col: 1,
                    rule,
                };
                kq.enqueue_gate_up_q4k(stream, &a, sink, &mut h)
                    .expect("the stack gate·up");
                let q = QuantSel {
                    x: &h,
                    cols: 0..1,
                    sel: &one,
                    n_card: 1,
                };
                gpu.q4k_sel()
                    .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                    .expect("the quantizer");
                let d = SelDown {
                    w: &stacks[PART_DOWN],
                    act: &act_h,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: N,
                };
                kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut y)
                    .expect("the stack down");
                (
                    h.to_host_vec(stream).expect("h"),
                    y.to_host_vec(stream).expect("y"),
                )
            })
            .collect();
        for r in 1..ROUNDS {
            assert_ne!(
                want[r],
                want[r - 1],
                "rounds {} and {r} have the same rows",
                r - 1
            );
        }

        // The tier: two paged layers, each part its own file at AT, each
        // window the part's aligned span.
        let mut windows = [0usize; PARTS];
        let mut slot_bytes = 0;
        for p in 0..PARTS {
            windows[p] = slot_bytes;
            slot_bytes += engram::direct::aligned_span(AT[p] as u64, len[p] as u64).1;
        }
        for direct in [false, true] {
            let tag = if direct { "refill-direct" } else { "refill" };
            let scratch = Scratch::new(tag);
            let paths: [_; PARTS] = std::array::from_fn(|p| scratch.0.join(format!("part{p}")));
            let split = split_in(&scratch);
            let layer = || {
                let mut arena = arena_of(8, 0);
                arena.windows = windows;
                arena.slot_bytes = slot_bytes;
                arena.parts =
                    vec![
                        std::array::from_fn(|p| one_run(p, AT[p] as u64, len[p], AT[p] % PAGE));
                        8
                    ];
                arena.part_of.set([0, 1, 2]).unwrap();
                Some(arena)
            };
            let source = crate::model::PinnedSource(ctx);
            let pinned =
                crate::model::arena_kind_of(&source, bloomery_levers::NvTierMem::Pinned).unwrap();
            let slots = [slot_bytes; 2];
            let grant = ArenaGrant::whole((2 * slots.iter().sum::<usize>()) as u64);
            let mut tier =
                NvTier::carve(vec![layer(), layer()], &slots, 2, grant, 0, &split, pinned)
                    .expect("the tier's carve");
            let dev0 = tier
                .mem
                .device_start()
                .expect("a pinned arena's card address");
            // SAFETY: the tier's whole mapping, live until `tier` drops after
            // the window and every launch that read it.
            let arena = unsafe { window::<u32>(dev0, tier.mem.len() / 4, ctx) };
            let mut offs = DeviceBuffer::<u64>::zeroed(stream, PARTS).expect("the table");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, FF).expect("h");
            let mut act_h = Q8Act::with_slots(stream, 1, FF).expect("act_h");
            let mut y = DeviceBuffer::<f32>::zeroed(stream, N).expect("y");
            let mut graph: Option<Graph> = None;
            let mut table: Option<Vec<u64>> = None;
            for round in 0..ROUNDS {
                let case = format!("{tag} round {round}");
                for (p, path) in paths.iter().enumerate() {
                    let mut file = vec![0xA5u8; AT[p]];
                    file.extend(expert(round, p));
                    file.resize(file.len().next_multiple_of(PAGE), 0x5A);
                    std::fs::write(path, file).unwrap();
                }
                let arena_l = tier.by_layer[LAYER].as_mut().unwrap();
                arena_l.files = shards_of(paths.iter().map(|path| {
                    Arc::new(if direct {
                        DirectFile::open(path).expect("a disk-backed file opens O_DIRECT")
                    } else {
                        DirectFile::open_buffered(path).unwrap()
                    })
                }));
                let pick = super::Pick {
                    slot: 1,
                    id: ID,
                    gone: None,
                };
                super::read_pick(&tier.mem, &super::read_of(arena_l, &pick))
                    .unwrap_or_else(|e| panic!("{case}: the slot's fill: {e}"));
                arena_l.of_id[ID as usize].store(2, Relaxed);
                arena_l.slots[1].state.store(FILLED, Relaxed);
                let pin = tier.pin(LAYER, &[ID]).unwrap();
                let t: Vec<u64> = (0..PARTS)
                    .map(|p| {
                        let at = pin.device_slot(ID, p).expect("a pinned slot's address") - dev0;
                        assert_eq!(at % 4, 0, "{case}: part {p} starts off a word");
                        at / 4
                    })
                    .collect();
                if let Some(t0) = &table {
                    assert_eq!(&t, t0, "{case}: the slot moved");
                } else {
                    offs.copy_from_host(stream, &t).expect("the table");
                    table = Some(t);
                }
                let run = |s: &cuda_core::CudaStream,
                           h: &mut DeviceBuffer<f32>,
                           act_h: &mut Q8Act,
                           y: &mut DeviceBuffer<f32>|
                 -> Result<(), crate::GpuError> {
                    let a = GateUpAt {
                        arena: &arena,
                        offs: &offs,
                        act: &act_x,
                        n_slots: 1,
                        rows_per_expert: FF,
                        slots_per_col: 1,
                        rule,
                    };
                    kq.enqueue_gate_up_q4k_at(s, &a, sink, h)?;
                    let q = QuantSel {
                        x: h,
                        cols: 0..1,
                        sel: &one,
                        n_card: 1,
                    };
                    gpu.q4k_sel().enqueue_quantize_sel(s, &q, sink, act_h)?;
                    let d = SelDownAt {
                        arena: &arena,
                        offs: &offs,
                        act: act_h,
                        n_slots: 1,
                        rows_per_expert: N,
                    };
                    kq.enqueue_gemv_q5k_sel_at(s, &d, sink, y)
                };
                for how in ["eager", "graph"] {
                    h.copy_from_host(stream, &[f32::NAN; FF]).expect("h");
                    y.copy_from_host(stream, &[f32::NAN; N]).expect("y");
                    if how == "eager" {
                        run(stream, &mut h, &mut act_h, &mut y).expect("the arena entries");
                    } else {
                        if graph.is_none() {
                            graph = Some(
                                Graph::capture(stream, |s| run(s, &mut h, &mut act_h, &mut y))
                                    .expect("the capture"),
                            );
                        }
                        graph.as_ref().unwrap().launch(stream).expect("the replay");
                    }
                    let got = (
                        h.to_host_vec(stream).expect("h"),
                        y.to_host_vec(stream).expect("y"),
                    );
                    assert!(
                        gpu.take_fault().expect("the fault word").is_none(),
                        "{case} {how}: a fault"
                    );
                    if round > 0 {
                        assert_ne!(
                            got,
                            want[round - 1],
                            "{case} {how}: the slot still reads the round before's expert"
                        );
                    }
                    for (what, g, w) in [
                        ("h", &got.0, &want[round].0),
                        ("down", &got.1, &want[round].1),
                    ] {
                        if let Some(i) = (0..w.len()).find(|&i| g[i].to_bits() != w[i].to_bits()) {
                            panic!(
                                "{case} {how}: {what} row {i}: arena {:#010x}, stack {:#010x}",
                                g[i].to_bits(),
                                w[i].to_bits()
                            );
                        }
                    }
                }
                drop(pin);
            }
            drop(graph.take());
            drop(std::mem::ManuallyDrop::into_inner(arena).into_raw_parts());
        }
    }

    /// The card leg ([`super::super::arena_leg::ArenaLeg`]) adds a row's held
    /// ids to its sum bit for bit as the stack launches do: two ids of a
    /// pinned tier held for the card in slots 0 and 3 of six, the others the
    /// host's, on row 1 of a two-row boundary whose sum starts nonzero; the
    /// reference runs the leg's q8_1 of the activation, the stack `_sel`
    /// gate·up, the slots' q8_1 and the stack `_sel` down over the same
    /// experts, their weighted sum and its add to the starting sum.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_card_leg_sums_its_held_ids_as_the_stack_launches_do() {
        use super::super::arena_leg::{ArenaLeg, DownAtEntry, GateUpAtEntry, LegLayer};
        use super::super::page::Word;
        use super::super::slots::HOST;
        use super::super::step::{Boundary, BoundaryShape};
        use super::NvTier;
        use crate::Gpu;
        use crate::card_leg::tests::{stack_bytes, words_of};
        use crate::card_slots::{CardAcc, CardSlotKernels};
        use crate::elem::ElemKernels;
        use crate::kquant::{Act, GateUpAct, PART_DOWN, PART_GATE, PART_UP, SelDown};
        use crate::q4k_sel::QuantSel;
        use crate::quant_sel::QuantSelKernels;
        use crate::tensor::{DeviceTensor, Q8Act};
        use cuda_core::DeviceBuffer;
        use engram::direct::DirectFile;
        use gguf::GgmlType;
        use model::placement::workstation::ArenaGrant;
        use std::sync::Arc;
        use std::sync::atomic::Ordering::Relaxed;
        const PAGE: usize = DIRECT_ALIGN;
        const LAYER: usize = 1;
        const N_EXPERT: u32 = 8;
        const ROW: usize = 1;
        // The model width and the routed width: two and one super-blocks.
        const N: usize = 512;
        const FF: usize = 256;
        const N_USED: usize = 6;
        // Per slot, the id the card holds, or HOST for a slot the host computes.
        const IDS: [u32; N_USED] = [2, HOST, HOST, 5, HOST, HOST];
        // Per part (gate, up, down): its type, rows and k.
        const ROLES: [(GgmlType, usize, usize); PARTS] = [
            (GgmlType::Q4_K, FF, N),
            (GgmlType::Q4_K, FF, N),
            (GgmlType::Q5_K, N, FF),
        ];
        // Each part's expert 0 first byte in its file: on a word, off the page.
        const AT: [usize; PARTS] = [1248, 36, 4000];
        let expert = |id: u32, p: usize| {
            stack_bytes(
                ROLES[p].0,
                ROLES[p].1,
                ROLES[p].2,
                (id as usize * PARTS + p) as u64 + 0x1eaf,
            )
        };
        let len: [usize; PARTS] = std::array::from_fn(|p| expert(0, p).len());
        let at = |id: u32, p: usize| AT[p] + id as usize * len[p];
        let wave = |n: usize, mul: usize, off: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * mul) % 2001) as f32 / 1000.0 - off)
                .collect()
        };
        let x = wave(N, 2654435761, 1.0);
        let hsum = wave(N, 40503, 0.5);
        let weights: Vec<f32> = (0..N_USED).map(|s| 0.125 + s as f32 / 16.0).collect();
        let places = IDS.map(|id| if id == HOST { HOST } else { 0 });
        let rule = Act::SwigluClamp { limit: 0.25 };
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let sink = gpu.layer_sink(LAYER).expect("a sink");

        // The reference: every expert as a stack row, the stack launches.
        let (want, leg_bytes) = {
            let kq = QuantSelKernels::load(ctx, gpu.fault_word()).expect("the kquant module");
            let card = CardSlotKernels::load(ctx).expect("the card slot module");
            let elem = ElemKernels::load(ctx, gpu.fault_word()).expect("the elem module");
            let stacks: Vec<DeviceTensor<u32>> = (0..PARTS)
                .map(|p| {
                    let rows = N_EXPERT as usize * ROLES[p].1;
                    let bytes: Vec<u8> = (0..N_EXPERT).flat_map(|id| expert(id, p)).collect();
                    let words = words_of(&bytes, rows);
                    let n = words.len();
                    DeviceTensor::upload(stream, &words, rows, n / rows).expect("a stack")
                })
                .collect();
            let x = DeviceBuffer::from_host(stream, &x).expect("x");
            let hsum = DeviceBuffer::from_host(stream, &hsum).expect("the sum");
            let w = DeviceBuffer::from_host(stream, &weights).expect("the weights");
            let ids = DeviceBuffer::from_host(stream, &IDS).expect("the ids");
            let places = DeviceBuffer::from_host(stream, &places).expect("the places");
            let x_place = DeviceBuffer::from_host(stream, &[0u32]).expect("x's place");
            let mut act_x = Q8Act::with_k(stream, 1, N).expect("act_x");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, N_USED * FF).expect("h");
            let mut act_h = Q8Act::with_slots(stream, N_USED, FF).expect("act_h");
            let mut down = DeviceBuffer::<f32>::zeroed(stream, N_USED * N).expect("down");
            let mut acc = DeviceBuffer::<f32>::zeroed(stream, N).expect("acc");
            let mut out = DeviceBuffer::<f32>::zeroed(stream, N).expect("out");
            let q = QuantSel {
                x: &x,
                cols: 0..1,
                sel: &x_place,
                n_card: 1,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_x)
                .expect("the activation's q8_1");
            let a = GateUpAct {
                wg: &stacks[PART_GATE],
                wu: &stacks[PART_UP],
                act: &act_x,
                sel: &ids,
                n_slots: N_USED,
                rows_per_expert: FF,
                slots_per_col: N_USED,
                rule,
            };
            kq.enqueue_gate_up_q4k(stream, &a, sink, &mut h)
                .expect("the stack gate·up");
            let q = QuantSel {
                x: &h,
                cols: 0..N_USED,
                sel: &places,
                n_card: 1,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                .expect("the slots' q8_1");
            let d = SelDown {
                w: &stacks[PART_DOWN],
                act: &act_h,
                sel: &ids,
                n_slots: N_USED,
                rows_per_expert: N,
            };
            kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut down)
                .expect("the stack down");
            let c = CardAcc {
                down: &down,
                w: &w,
                sel: &places,
                n: N,
                m: 1,
                n_card: 1,
                n_used: N_USED,
                w_pitch: N_USED,
                fault: sink,
            };
            card.enqueue_card_acc(stream, &c, &mut acc)
                .expect("the weighted sum");
            elem.enqueue_add(stream, &hsum, &acc, N, &mut out)
                .expect("the add");
            let leg_bytes = x_place.num_bytes()
                + act_x.device_bytes()
                + h.num_bytes()
                + act_h.device_bytes()
                + down.num_bytes()
                + acc.num_bytes();
            (out.to_host_vec(stream).expect("the sum"), leg_bytes)
        };
        assert_ne!(want, hsum, "the held ids add nothing to the sum");

        // The tier: one paged layer of two slots, each part its own file
        // holding every expert from AT, each window the widest span an
        // expert's part takes.
        let scratch = Scratch::new("card-leg-sum");
        let split = split_in(&scratch);
        let files: [_; PARTS] = std::array::from_fn(|p| {
            let path = scratch.0.join(format!("part{p}"));
            let mut bytes = vec![0xA5u8; AT[p]];
            bytes.extend((0..N_EXPERT).flat_map(|id| expert(id, p)));
            bytes.resize(bytes.len().next_multiple_of(PAGE), 0x5A);
            std::fs::write(&path, bytes).unwrap();
            Arc::new(DirectFile::open_buffered(&path).unwrap())
        });
        let mut windows = [0usize; PARTS];
        let mut slot_bytes = 0;
        for p in 0..PARTS {
            windows[p] = slot_bytes;
            slot_bytes += (0..N_EXPERT)
                .map(|id| aligned_span(at(id, p) as u64, len[p] as u64).1)
                .max()
                .unwrap();
        }
        let mut arena = arena_of(N_EXPERT, 0);
        arena.windows = windows;
        arena.slot_bytes = slot_bytes;
        arena.parts = (0..N_EXPERT)
            .map(|id| {
                std::array::from_fn(|p| one_run(p, at(id, p) as u64, len[p], at(id, p) % PAGE))
            })
            .collect();
        arena.files = shards_of(files);
        arena.part_of.set([0, 1, 2]).unwrap();
        let source = crate::model::PinnedSource(ctx);
        let pinned =
            crate::model::arena_kind_of(&source, bloomery_levers::NvTierMem::Pinned).unwrap();
        let grant = ArenaGrant::whole((2 * slot_bytes) as u64);
        let tier = Arc::new(
            NvTier::carve(
                vec![None, Some(arena)],
                &[slot_bytes],
                2,
                grant,
                0,
                &split,
                pinned,
            )
            .expect("the tier's carve"),
        );
        assert!(tier.parts_on_words(LAYER), "every part on a word");

        // The leg on row 1, its slots written as a service writes them.
        let shape = BoundaryShape {
            hidden: N,
            n_used: N_USED,
        };
        let mut boundary = Boundary::with_rows(ctx, stream, shape, 2).expect("a two-row boundary");
        let boundary_bytes = boundary.device_bytes();
        let h = boundary.layout().handoff();
        let (image, hsum_off) = boundary.row_offsets(ROW).unwrap();
        let put = |off: usize, v: &[f32]| {
            for (i, f) in v.iter().enumerate() {
                boundary
                    .page()
                    .atomic_u32(off + 4 * i)
                    .expect("a word of the page")
                    .store(f.to_bits(), Relaxed);
            }
        };
        put(image + 4 * h.x, &x);
        put(image + 4 * h.weights, &weights);
        put(hsum_off, &hsum);
        let mut layers = vec![None; LAYER + 1];
        layers[LAYER] = Some(LegLayer {
            gate_up: GateUpAtEntry::Q4k,
            down: DownAtEntry::Q5k,
            rule,
            fault: sink,
        });
        let mut leg = ArenaLeg::build(&gpu, &boundary, &tier, layers, FF).expect("the leg");
        for error in [
            leg.skip_row(2).unwrap_err(),
            leg.set_slot(2, 0, None).unwrap_err(),
            leg.set_slot(ROW, N_USED, None).unwrap_err(),
            leg.enqueue(stream, 2, LAYER).unwrap_err(),
        ] {
            let text = error.to_string();
            assert!(
                text.contains("row 2 of 2") || text.contains("slot 6 of 6"),
                "{text}"
            );
        }
        leg.skip_row(ROW).expect("a row inside the leg");
        let held: Vec<u32> = IDS.into_iter().filter(|&id| id != HOST).collect();
        let hold = tier.hold_for_card(LAYER, &held).expect("the hold");
        for (s, &id) in IDS.iter().enumerate() {
            if id != HOST {
                let offs = hold.offsets(id).expect("a held id's offsets");
                leg.set_slot(ROW, s, Some(offs))
                    .expect("a slot inside the leg");
            }
        }
        boundary.arena = Some(leg);
        assert_eq!(
            boundary.device_bytes(),
            boundary_bytes + leg_bytes,
            "the boundary counts the leg's owned device buffers"
        );
        boundary
            .page()
            .atomic_u32(Word::Cnt(ROW).offset())
            .expect("the row's counter")
            .store(1, std::sync::atomic::Ordering::Release);
        boundary
            .enqueue_back_of(stream, ROW, LAYER)
            .expect("the row's card leg");
        stream.synchronize().expect("the stream");
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "the leg raised a fault"
        );
        let got = boundary
            .page()
            .f32_copy(hsum_off, N)
            .expect("the row's sum");
        if let Some(i) = (0..N).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            panic!(
                "the card leg's sum at {i}: {:#010x}, the stack launches' {:#010x}",
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        drop(hold);
        drop(boundary);
    }

    /// A pin names what it refuses and pins nothing then: an id with no
    /// filled slot, and an id past the experts; an id of the host segment is
    /// the mapping's and pins nothing.
    #[test]
    fn a_pin_of_an_unfilled_or_unknown_id_is_refused_and_pins_nothing() {
        let mut arena = arena_of(8, 2);
        arena.host = 6..8;
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0, 1], Claim::Step);
        match arena.pin_batch(7, &[0, 5], "test") {
            Err(crate::GpuError::Protocol { what, detail }) => {
                assert_eq!(what, "test");
                assert!(
                    detail.contains("layer 7 expert 5") && detail.contains("unfilled"),
                    "{detail}"
                );
            }
            other => panic!("an unfilled id is refused, got {:?}", other.map(|_| ())),
        }
        assert!(
            arena.pin_batch(7, &[0, 99], "test").is_err(),
            "an id past the experts"
        );
        let open = |id| {
            arena
                .reading(7, id, "test")
                .unwrap()
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        assert_eq!((open(0), open(1)), (0, 0), "the refusals moved no count");
        let pin = arena
            .pin_batch(7, &[6, 7], "test")
            .expect("host ids pin nothing");
        assert_eq!((open(6), open(7)), (0, 0));
        drop(pin);
    }

    /// A batch pin counts as a lane's read of a filled slot: beside a lane's
    /// read of the same id, opened filled or opened unfilled (the mapping's)
    /// and filled since, the open reads classify as slot reads without a
    /// refusal, and a pin alone is never read as a mapping read.
    #[test]
    fn a_lane_read_and_a_batch_pin_on_one_id_read_as_a_slot() {
        use std::sync::atomic::Ordering::Relaxed;
        let arena = arena_of(4, 2);
        let mut books = super::Books::default();
        ensure_as(&arena, &mut books, &[0], Claim::Step);
        arena
            .open(0, 0, "test")
            .expect("a lane opens the filled id");
        let pin = arena.pin_batch(0, &[0], "test").expect("a batch pin");
        assert!(arena.slot_read(0, 0, "test").unwrap(), "a slot read");
        drop(pin);
        arena.close(0, 0, "test").expect("the lane's end");
        // A lane opens id 1 unfilled: the mapping's read. A fill and a pin
        // follow: the open reads are no longer all the mapping's.
        arena
            .open(0, 1, "test")
            .expect("a lane opens the unfilled id");
        arena.mapped[1].fetch_add(1, Relaxed);
        assert!(
            !arena.slot_read(0, 1, "test").unwrap(),
            "the mapping's read"
        );
        ensure_as(&arena, &mut books, &[1], Claim::Step);
        let pin = arena.pin_batch(0, &[1], "test").expect("a batch pin");
        assert!(
            arena.slot_read(0, 1, "test").unwrap(),
            "the slot is filled and pinned"
        );
        drop(pin);
        assert!(
            !arena.slot_read(0, 1, "test").unwrap(),
            "the mapping's read again"
        );
        assert!(arena.slot_read(0, 2, "test").is_err(), "no read open");
    }

    /// Disjoint arena and mapping ranges pass; an arena range that names a
    /// model page — the mutant — does not.
    #[test]
    fn the_arena_never_advises_the_model_mapping() {
        let model = [(0x1000_0000, 0x2000_0000), (0x3000_0000, 0x4000_0000)];
        assert!(advice_disjoint(
            &[(0x0000_1000, 0x1000_0000), (0x2000_0000, 0x3000_0000)],
            &model
        ));
        assert!(!advice_disjoint(&[(0x1800_0000, 0x2800_0000)], &model));
        assert!(!advice_disjoint(&[(0x0, 0x9000_0000)], &model));
        assert!(advice_disjoint(&[(0x2000_0000, 0x3000_0000)], &model));
    }

    /// The audit's sites touch the span's two edge pages, its own ends and
    /// a middle page: whatever a torn fill leaves, one site reads it.
    #[test]
    fn audit_sites_cover_both_edge_pages_and_the_middle() {
        let page = DIRECT_ALIGN;
        for (at, len) in [(0, page), (7 * page + 13, 3 * page), (page + 1, 2 * page)] {
            let sites = audit_offsets(at as u64, len);
            assert!(sites.iter().all(|&s| s < len), "{at}+{len}: {sites:?}");
            assert!(sites.contains(&0) && sites.contains(&(len - 1)));
            assert!(sites.iter().any(|&s| s < page), "{at}+{len}: {sites:?}");
            assert!(
                sites.iter().any(|&s| s + page >= len),
                "{at}+{len}: {sites:?}"
            );
        }
    }

    /// A part of one run, `len` bytes at file offset `at` of shard `shard`,
    /// its aligned read landing `skew` bytes into its window.
    fn one_run(shard: usize, at: u64, len: usize, skew: usize) -> PartRange {
        PartRange {
            codes: super::Run { shard, at, len },
            scales: None,
            skew,
        }
    }

    /// The fills' handles of an arena whose shard `s` is `files`' `s`-th.
    fn shards_of(
        files: impl IntoIterator<Item = std::sync::Arc<engram::direct::DirectFile>>,
    ) -> std::sync::Arc<[Option<std::sync::Arc<engram::direct::DirectFile>>]> {
        files.into_iter().map(Some).collect()
    }

    /// A one-layer arena of `slots` slots over `n_expert` experts and no host
    /// segment, its one file a scratch file the books never read.
    fn arena_of(n_expert: u32, slots: usize) -> super::LayerArena {
        let path = std::env::temp_dir().join(format!("nvtier-books-{}", std::process::id()));
        std::fs::write(&path, [0u8; 16]).unwrap();
        let file = std::sync::Arc::new(engram::direct::DirectFile::open_buffered(&path).unwrap());
        let part = one_run(0, 0, 1, 0);
        super::LayerArena {
            host: 0..0,
            n_expert,
            rows: PARTS,
            windows: [0; PARTS],
            slot_bytes: DIRECT_ALIGN,
            base: 0,
            dev: None,
            part_of: std::sync::OnceLock::new(),
            files: shards_of([file]),
            names: Default::default(),
            parts: vec![[part; PARTS]; n_expert as usize],
            of_id: (0..n_expert).map(|_| AtomicU32::new(0)).collect(),
            reading: (0..n_expert).map(|_| AtomicU32::new(0)).collect(),
            mapped: (0..n_expert).map(|_| AtomicU32::new(0)).collect(),
            slots: (0..slots).map(|_| book(FREE, 0)).collect(),
            idle: super::id_pool(slots),
        }
    }

    /// Mark what a call claimed as filled, as the fill's join does.
    fn fill_claimed(arena: &super::LayerArena, books: &mut super::Books, c: &super::Claimed) {
        fill_claimed_as(arena, books, c, Claim::Step);
    }

    /// [`fill_claimed`] for a call claimed as `kind`.
    fn fill_claimed_as(
        arena: &super::LayerArena,
        books: &mut super::Books,
        c: &super::Claimed,
        kind: Claim,
    ) {
        super::publish(arena, books, &c.picks, kind);
    }

    /// A layer's drop keeps the pages of an id a lane has open: its mark is
    /// held while the read is open and read again once it closes, the host
    /// segment's held throughout; an end with no open is refused by name,
    /// never a count wrapped past zero, and so is an id past the experts. A
    /// layer's drop that read the books without the open counts would name
    /// the open id's pages.
    #[test]
    fn an_open_read_keeps_its_id_out_of_the_layers_drop() {
        use model::placement::paged_drop::Mark::{Held, Read};
        let mut arena = arena_of(4, 1);
        arena.host = 0..1;
        assert_eq!(arena.layer_marks(), vec![Held, Read, Read, Read]);
        arena.open(7, 2, "test").expect("an id of the layer");
        arena.open(7, 2, "test").expect("a second lane on the id");
        assert_eq!(arena.layer_marks(), vec![Held, Read, Held, Read]);
        arena.close(7, 2, "test").expect("the first lane's end");
        assert_eq!(arena.layer_marks()[2], Held, "the second lane reads on");
        arena.close(7, 2, "test").expect("the second lane's end");
        assert_eq!(arena.layer_marks(), vec![Held, Read, Read, Read]);
        match arena.close(7, 2, "test") {
            Err(crate::GpuError::Protocol { what, detail }) => {
                assert_eq!(what, "test");
                assert!(detail.contains("layer 7 expert 2"), "{detail}");
            }
            other => panic!("an end with no open is refused, got {other:?}"),
        }
        assert_eq!(arena.layer_marks()[2], Read, "the refusal moved no count");
        assert!(arena.open(7, 4, "test").is_err(), "an id past the experts");
    }

    /// Two calls on one layer with an eviction between them: the second
    /// refills the evicted id and its slot reads filled. A `seen` mark that
    /// outlived the first call would skip it and leave the slot unfilled.
    #[test]
    fn an_evicted_id_is_refilled_by_a_later_call() {
        let arena = arena_of(4, 2);
        let mut books = super::Books::default();
        let first = super::claim_slots(0, &arena, &mut books, &[0, 1], Claim::Step).unwrap();
        assert_eq!((first.picks.len(), first.misses), (2, 2));
        fill_claimed(&arena, &mut books, &first);
        // Expert 2 takes the slot of the least recently used, expert 0.
        let evict = super::claim_slots(0, &arena, &mut books, &[2], Claim::Step).unwrap();
        assert_eq!(evict.evicted.len(), 1);
        fill_claimed(&arena, &mut books, &evict);
        assert!(arena.filled(0).is_none(), "expert 0 was evicted");
        // The call that needs expert 0 again claims a slot for it.
        let again = super::claim_slots(0, &arena, &mut books, &[0], Claim::Step).unwrap();
        assert_eq!(again.picks.len(), 1, "an evicted id is refilled");
        fill_claimed(&arena, &mut books, &again);
        assert!(arena.filled(0).is_some(), "its slot reads filled");
    }

    /// A call naming more distinct ids of the arena than the layer's slots
    /// is refused by name with the layer, the count asked and the slot
    /// count, before any slot is claimed — not the race refusal a failed
    /// pick would name — and the books and slots stand as they were; the
    /// seam passes the refusal on whole. The race refusal keeps its own
    /// case: a miss with every slot filling.
    #[test]
    fn a_call_past_the_slots_is_refused_before_any_claim() {
        use model::moe::TierError;
        let arena = arena_of(4, 1);
        let mut books = super::Books::default();
        // One slot, id 1 already filled; the call names three distinct ids
        // of the arena (id 2 twice, counted once), one of them the filled
        // one — the union reads them all at once.
        let first = super::claim_slots(7, &arena, &mut books, &[1], Claim::Step).unwrap();
        fill_claimed(&arena, &mut books, &first);
        let tick = books.tick;
        let e = super::claim_slots(7, &arena, &mut books, &[1, 0, 2, 2], Claim::Step)
            .map(|_| ())
            .unwrap_err();
        match &e {
            crate::GpuError::Model(m) => assert!(
                matches!(
                    **m,
                    model::ModelError::Tier(TierError::TooManyIds {
                        layer: 7,
                        ids: 3,
                        slots: 1
                    })
                ),
                "{e}"
            ),
            other => panic!("the seam's named refusal, got {other}"),
        }
        let text = e.to_string();
        assert!(
            text.contains("layer 7") && text.contains("3 distinct ids") && text.contains("1 slots"),
            "{text}"
        );
        assert_eq!(arena.filled(1), Some(0), "the filled id's book stands");
        assert_eq!(arena.filled(0), None, "no id was claimed");
        assert_eq!(books.tick, tick, "the books' tick stands");
        assert_eq!(
            arena.slots[0]
                .state
                .load(std::sync::atomic::Ordering::Relaxed),
            FILLED,
            "the slot is not left filling"
        );
        // The seam hands the model side the refusal itself, not a fill's
        // failure of the call's first id.
        assert!(
            matches!(
                super::seam_error(7, &[1, 0, 2, 2], e),
                TierError::TooManyIds {
                    layer: 7,
                    ids: 3,
                    slots: 1
                }
            ),
            "the seam passes the books' refusal on whole"
        );

        // The race refusal keeps its case: a miss with every slot filling.
        arena.slots[0]
            .state
            .store(FILLING, std::sync::atomic::Ordering::Relaxed);
        let raced = super::claim_slots(7, &arena, &mut books, &[3], Claim::Step)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            raced.contains("raced") && raced.contains("filling"),
            "{raced}"
        );
    }

    /// Layers' slots sit one after the other in the mapping: no two layers'
    /// slot ranges overlap and the last ends at `count × Σ slot sizes`.
    #[test]
    fn layers_slots_never_overlap() {
        let sizes = [24_576usize, 28_672, 24_576];
        let count = 5;
        let bases = layer_bases(count, &sizes);
        let ranges: Vec<(usize, usize)> = bases
            .iter()
            .zip(sizes)
            .map(|(&b, s)| (b, b + count * s))
            .collect();
        for (i, a) in ranges.iter().enumerate() {
            for b in &ranges[i + 1..] {
                assert!(a.1 <= b.0 || b.1 <= a.0, "{a:?} overlaps {b:?}");
            }
        }
        assert_eq!(
            ranges.last().map(|r| r.1),
            Some(count * sizes.iter().sum::<usize>())
        );
    }

    /// A slot's part windows sit at their own aligned bases, one part never
    /// crossing the next, and the logical bytes at the skew inside the
    /// window the fill wrote.
    #[test]
    fn a_slots_windows_are_aligned_and_disjoint() {
        let a = DIRECT_ALIGN;
        let lens = [3 * a + 512, 3 * a + 512, 2 * a + 100];
        let mut windows = [0usize; PARTS];
        for p in 1..PARTS {
            windows[p] = windows[p - 1] + window_bytes(lens[p - 1], 0);
        }
        let slot_bytes = windows[PARTS - 1] + window_bytes(lens[PARTS - 1], 0);
        assert_eq!(slot_bytes % a, 0, "the slot is aligned");
        for (p, w) in windows.iter().enumerate() {
            assert_eq!(w % a, 0, "part {p}'s window is aligned");
        }
        for slot in [0usize, 1, 7] {
            for p in 0..PARTS {
                let r = one_run(0, 0, lens[p], a - 1);
                let (at, len) = window_span(slot * slot_bytes, &windows, p, r);
                assert_eq!(at % a, r.skew % a + (windows[p] % a));
                assert_eq!(len, lens[p]);
                assert!(
                    at + len <= (slot + 1) * slot_bytes,
                    "part {p} of slot {slot} crosses the slot"
                );
            }
        }
    }
}
