//! The NVMe expert tier's RAM arena: one anonymous mapping
//! ([`engram::direct::AnonMap`]) carved into slots — one expert a slot,
//! each of its parts in its own [`DIRECT_ALIGN`]-aligned window, the bytes
//! landing at the skew an aligned read of the part's file range needs —
//! filled on a miss by [`DirectFile::read_span`] over the resident pool,
//! and lent to the host leg through the model side's [`TierSlots`] seam.
//! The file mapping serves the plan's host segment; every other id the leg
//! reads — the NVMe segment's, and the card's victims once they flip — is
//! this arena's, its slot refused by name while unfilled.
//!
//! The arena's own pages leave on an eviction alone, `MADV_DONTNEED` on the
//! evicted slot's windows, which frees an anonymous range's pages — that is
//! why the arena is anonymous and not the page cache, whose warm set nothing
//! else can flush. The model file's mapping is its readers': a read of the
//! ids the arena serves through it (a prompt call's union, a lane's copy)
//! faults their pages in from the drive, and once the reader has consumed
//! them the tier drops them from the mapping and the page cache — a prompt
//! union's whole layer ([`NvTier::release_union`]) but the ids a lane has
//! open, a lane's own id ([`NvTier::end_read`]) — so nothing a prompt or a
//! lane brought in stays past it. A decode step's union of several columns
//! (the step port) keeps what it read: its ids are the decode's hot set,
//! which the next step reads again, and a drop would send each step back to
//! the drive for them. Which pages go is the books' ([`drop_runs`]), never `mincore`'s, and no
//! page that holds a byte of the plan's host segment, of an id a lane has
//! open, or of the tensors beside a stack is named. The books
//! are one atomic hint an id and a per-slot state word a pick reads without
//! the lock; the LRU and the fills' slot ownership live under one lock
//! taken per `ensure`, never per pick. The engine serves a model's host leg
//! one service at a time, so no pick reads a slot another service evicts —
//! the seat and the run-ahead that share the arena across threads join
//! their fills before the books give the slot away.

use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use engram::direct::{AnonMap, DIRECT_ALIGN, DirectFile, aligned_span};
use gguf::Split;
use model::moe::{TierError, TierSlots};
use model::placement::host_lock::{HostFile, HostSet, PageDrop, page_bytes};
use model::placement::paged_drop::{Mark, drop_runs, overlap_of};
use model::placement::{Device, Plan, Role};

use crate::GpuError;

/// A slot's state, as a pick reads it without the lock.
const FREE: u32 = 0;
const FILLING: u32 = 1;
const FILLED: u32 = 2;

/// The parts of an expert the arena holds: a slot's windows in the plan's
/// stack-row order, and the count [`TierSlots::slot`]'s `part` names —
/// mapped to the rows per layer at the attach ([`NvTier::note_parts`]).
const PARTS: usize = 3;

/// One part of one expert as its file holds it: the byte range of the
/// matrix, and the skew its aligned read lands at in the part's window —
/// static per (layer, id, part), resolved at the build.
#[derive(Clone, Copy)]
struct PartRange {
    at: u64,
    len: usize,
    skew: usize,
}

/// The bytes a part's window takes in its slot: the part's own bytes and
/// the two partial pages an aligned read of them can span — the same shape
/// the staging swap's ring slot uses.
pub(crate) fn window_bytes(len: usize) -> usize {
    (len + 2 * DIRECT_ALIGN).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN
}

/// The slots a layer holds for the arena's `budget` bytes over the paged
/// layers' slot sizes: the same count a layer, so every layer warms alike
/// and the carved total stays under the budget. A budget under one slot of
/// every layer is the named refusal — the arena would hold nothing and
/// every read refuse.
pub(crate) fn slots_of(budget: u64, slot_bytes: &[usize]) -> Result<usize, String> {
    let total: usize = slot_bytes.iter().sum();
    if total == 0 {
        return Err("the plan pages no routed stack".to_string());
    }
    let n = (budget as usize) / total;
    if n == 0 {
        return Err(format!(
            "the arena's budget {budget} B is under one slot of every paged layer ({total} B)"
        ));
    }
    Ok(n)
}

/// The slot a miss takes: the first free one, else the least recently used
/// filled — a filling slot is never the arena's to take, its fill owns it.
/// `None` when every slot is filling (two ensures raced a layer).
fn pick_slot(slots: &[SlotBook]) -> Option<usize> {
    if let Some(k) = slots
        .iter()
        .position(|s| s.state.load(Ordering::Relaxed) == FREE)
    {
        return Some(k);
    }
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.state.load(Ordering::Relaxed) == FILLED)
        .min_by_key(|(_, s)| s.tick.load(Ordering::Relaxed))
        .map(|(k, _)| k)
}

/// What one `ensure` claimed under the books' lock: the slots to fill, the
/// filled slots it let go (their pages to return) and the ids it did not
/// find filled.
struct Claim {
    picks: Vec<Pick>,
    evicted: Vec<usize>,
    misses: u64,
}

/// The books' pass of one `ensure` over `layer`'s `ids`: ids the plan's host
/// segment holds are skipped, a filled one's tick is refreshed, every other
/// takes a slot — the first free, else the least recently used filled one,
/// which it evicts — and is marked filling. Pure over the books: no read, no
/// page returned. The `seen` marks are this call's alone; a mark that
/// outlived its call would skip an expert evicted since, and its slot would
/// read unfilled.
fn claim_slots(arena: &LayerArena, books: &mut Books, ids: &[u32]) -> Result<Claim, String> {
    if books.seen.len() != arena.n_expert as usize {
        books.seen.resize(arena.n_expert as usize, false);
    }
    books.seen.fill(false);
    let mut claim = Claim {
        picks: Vec::new(),
        evicted: Vec::new(),
        misses: 0,
    };
    for &id in ids {
        let Some(at) = usize::try_from(id)
            .ok()
            .filter(|&i| i < arena.n_expert as usize)
        else {
            return Err(format!(
                "expert {id} is past the {} experts",
                arena.n_expert
            ));
        };
        if arena.host.contains(&id) || std::mem::replace(&mut books.seen[at], true) {
            continue;
        }
        if let Some(slot) = arena.filled(id) {
            books.tick += 1;
            arena.slots[slot].tick.store(books.tick, Ordering::Relaxed);
            continue;
        }
        claim.misses += 1;
        let slot = pick_slot(&arena.slots)
            .ok_or("every slot is filling: two ensures raced a layer".to_string())?;
        if arena.slots[slot].state.load(Ordering::Relaxed) == FILLED {
            let gone = arena.slots[slot].id.load(Ordering::Relaxed);
            arena.of_id[gone as usize].store(0, Ordering::Relaxed);
            claim.evicted.push(slot);
        }
        arena.slots[slot].state.store(FILLING, Ordering::Relaxed);
        arena.slots[slot].id.store(id, Ordering::Relaxed);
        arena.of_id[at].store(slot as u32 + 1, Ordering::Relaxed);
        claim.picks.push(Pick { slot, id });
    }
    Ok(claim)
}

/// Where part `part` of the expert slot at byte `slot_off` of the mapping
/// holds `r`'s bytes: the logical bytes' offset in the mapping and their
/// length (the window's base plus the run's skew).
fn window_span(
    slot_off: usize,
    windows: &[usize; PARTS],
    part: usize,
    r: PartRange,
) -> (usize, usize) {
    (slot_off + windows[part] + r.skew, r.len)
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
    /// Slot `k`'s part `p` window at `windows[p]` bytes into the slot.
    windows: [usize; PARTS],
    slot_bytes: usize,
    /// The layer's first byte in the mapping ([`layer_bases`]).
    base: usize,
    /// The plan's stack row each of the leg's parts reads, resolved at the
    /// attach by name ([`NvTier::note_parts`]); unread until then, and
    /// every pick of the layer refused by name after.
    part_of: OnceLock<[usize; PARTS]>,
    /// The file each of the plan's stack rows reads.
    files: [Arc<DirectFile>; PARTS],
    /// The split's shard each of the plan's stack rows lies in, whose
    /// mapping a drop advises.
    shards: [usize; PARTS],
    /// The plan's stack-row tensors' names, for the attach's mapping.
    names: [String; PARTS],
    /// Per id: its parts' ranges, by stack row.
    parts: Vec<[PartRange; PARTS]>,
    /// Per id: the slot that holds it, + 1; 0 when none does.
    of_id: Vec<AtomicU32>,
    /// Per id: the lanes reading it through the model file's mapping now
    /// ([`NvTier::open_read`]); a drop keeps its pages while it is nonzero.
    reading: Vec<AtomicU32>,
    /// Per slot: its state (a pick's read), its id and LRU tick (the
    /// lock's alone).
    slots: Vec<SlotBook>,
}

impl LayerArena {
    /// Slot `slot`'s first byte in the mapping.
    fn slot_off(&self, slot: usize) -> usize {
        self.base + slot * self.slot_bytes
    }

    /// The slot holding expert `id` filled, if one does — the one reading
    /// of the books a pick, the residency seam and `ensure`'s hit check
    /// share. `id` past the layer's experts holds none.
    fn filled(&self, id: u32) -> Option<usize> {
        let held = self.of_id.get(id as usize)?.load(Ordering::Relaxed);
        let slot = (held as usize).checked_sub(1)?;
        (self.slots[slot].state.load(Ordering::Relaxed) == FILLED).then_some(slot)
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

    /// The expert whose bytes of stack row `row` hold byte `at` of the file,
    /// for a refusal to name: the experts' count for a byte past them.
    fn id_at(&self, row: usize, at: u64) -> usize {
        self.parts
            .partition_point(|p| p[row].at + p[row].len as u64 <= at)
    }
}

struct SlotBook {
    state: AtomicU32,
    /// The slot's expert, +1 semantics none: written under the books'
    /// lock, read by an evicting pass.
    id: AtomicU32,
    /// The slot's LRU tick: written under the books' lock, read by the
    /// victim pick.
    tick: AtomicU64,
}

/// What the books picked for one miss: the slot to fill and, when it held
/// an expert, the id it let go.
struct Pick {
    slot: usize,
    id: u32,
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
    drops: AtomicU64,
    drop_bytes: AtomicU64,
    drop_ns: AtomicU64,
}

/// The RAM arena of the NVMe expert tier ([`NvTier::of_paged`]).
pub struct NvTier {
    map: AnonMap,
    /// One arena a layer, indexed by the plan's layer; `None` on a layer
    /// the plan pages nothing of.
    by_layer: Vec<Option<LayerArena>>,
    /// The LRU tick and the dedup pass's marks, under one lock taken per
    /// `ensure`.
    books: Mutex<Books>,
    stats: Stats,
    budget: u64,
    /// The routed-expert bytes the plan pages on the NVMe tier.
    paged: u64,
    /// One buffered handle per shard, once [`NvTier::audit_on`] turned the
    /// audit on.
    audit: OnceLock<Vec<(PathBuf, std::fs::File)>>,
    /// The mapping's first byte, for the pool's fill workers — each fill
    /// writes its own slot's own window, disjoint by the books' slot
    /// ownership.
    shared: SharedArena,
    /// The split whose shards' mappings the readers read and the drops
    /// advise.
    split: Arc<Split>,
    /// The host's page in bytes, the unit a drop's runs round to.
    page: u64,
}

struct Books {
    tick: u64,
    /// The dedup pass's seen marks, reused across calls.
    seen: Vec<bool>,
}

/// The arena's mapping as the pool's fill workers reach it.
struct SharedArena(*mut u8);

// SAFETY: it moves between threads only inside the arena it belongs to.
unsafe impl Send for SharedArena {}
// SAFETY: the pointer is the arena's own mapping, written only through
// disjoint windows — one slot's one part a fill, the slot owned by exactly
// one in-flight fill through the books — while the `&self` that made it is
// held by `ensure`, which returns once every fill has joined.
unsafe impl Sync for SharedArena {}

impl NvTier {
    /// The arena of `plan`'s NVMe expert tier over `split`'s shards, read
    /// `O_DIRECT` (`direct`) or through the page cache: `None` on a plan
    /// that pages no routed expert or reserved no arena bytes
    /// (`HostTotals::nvme_arena_bytes` 0 — the mapping path as it stands).
    /// The slots' budget is the arena the dial reserved, the same count a
    /// paged layer. Refused by name: a layer whose host segment's ids are
    /// not one run, whose routed stacks are not three rows the split holds,
    /// a stack whose experts do not cut its bytes evenly, a stack in a shard
    /// that is not its file's mapping ([`gguf::Gguf::file_backed`]: a drop
    /// would zero a copy), and a budget under one slot.
    pub fn of_paged(
        plan: &Plan<'_>,
        split: &Arc<Split>,
        direct: bool,
    ) -> Result<Option<NvTier>, GpuError> {
        const WHAT: &str = "NvTier::of_paged";
        if plan.host.nvme_expert_bytes == 0 || plan.host.nvme_arena_bytes == 0 {
            return Ok(None);
        }
        let model = plan.model;
        // Per layer: its routed stack rows' tensor names in row order, and
        // its host segment's id run.
        let mut rows_of: Vec<Vec<String>> = (0..model.layers).map(|_| Vec::new()).collect();
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
            rows_of[l].push(t.name.clone());
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
        // The shards' direct handles, one a file.
        let mut files: Vec<Arc<DirectFile>> = Vec::new();
        let file_of =
            |files: &mut Vec<Arc<DirectFile>>, path: &std::path::Path| -> Result<usize, GpuError> {
                if let Some(i) = files.iter().position(|f| f.path() == path) {
                    return Ok(i);
                }
                let f = Arc::new(
                    if direct {
                        DirectFile::open(path)
                    } else {
                        DirectFile::open_buffered(path)
                    }
                    .map_err(|e| GpuError::plan(WHAT, e))?,
                );
                files.push(f);
                Ok(files.len() - 1)
            };
        let mut arenas: Vec<Option<LayerArena>> = (0..model.layers).map(|_| None).collect();
        let mut slot_sizes = Vec::new();
        for (l, rows) in rows_of.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            if rows.len() != PARTS {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l} holds {} routed stack rows, not {PARTS}",
                        rows.len()
                    ),
                ));
            }
            let n = model.experts as usize;
            let mut parts: Vec<Vec<PartRange>> = vec![Vec::new(); n];
            let mut lens = [0usize; PARTS];
            let mut rows_files = [0usize; PARTS];
            let mut rows_shards = [0usize; PARTS];
            for (p, name) in rows.iter().enumerate() {
                let (shard, info) = split
                    .find(name)
                    .ok_or(GpuError::shape(WHAT, format!("{name}: not in the split")))?;
                let gguf = split.shard(shard).ok_or(GpuError::shape(
                    WHAT,
                    format!("{name}: shard {shard} is not there"),
                ))?;
                if !gguf.file_backed() {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "{name}: shard {shard} is an anonymous copy, not its file's mapping: \
                             the tier drops a paged stack's pages after each read, which would \
                             zero the copy"
                        ),
                    ));
                }
                rows_shards[p] = shard;
                let path = split.shard_path(shard).ok_or(GpuError::shape(
                    WHAT,
                    format!("{name}: shard {shard} has no path"),
                ))?;
                rows_files[p] = file_of(&mut files, path)?;
                let Some(&stack) = info.dims.last() else {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{name}: no expert count in its dims"),
                    ));
                };
                if stack as usize != n {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{name}: {stack} experts, the model has {n}"),
                    ));
                }
                let per = usize::try_from(info.nbytes / stack).map_err(|_| {
                    GpuError::shape(WHAT, format!("{name}: one expert's bytes pass usize"))
                })?;
                if per * n != info.nbytes as usize {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{name}: its experts do not cut its bytes evenly"),
                    ));
                }
                lens[p] = per;
                let base = gguf.data_base() + info.offset;
                for id in 0..n as u32 {
                    let at = base + u64::from(id) * per as u64;
                    let (_, _, need) = aligned_span(at, per as u64);
                    parts[id as usize].push(PartRange {
                        at,
                        len: per,
                        skew: need - per,
                    });
                }
            }
            let mut windows = [0usize; PARTS];
            for p in 1..PARTS {
                windows[p] = windows[p - 1] + window_bytes(lens[p - 1]);
            }
            let slot_bytes = windows[PARTS - 1] + window_bytes(lens[PARTS - 1]);
            slot_sizes.push(slot_bytes);
            arenas[l] = Some(LayerArena {
                host: host_of[l].clone().unwrap_or(0..0),
                n_expert: model.experts as u32,
                windows,
                slot_bytes,
                base: 0,
                part_of: OnceLock::new(),
                files: rows_files.map(|i| Arc::clone(&files[i])),
                shards: rows_shards,
                names: rows.clone().try_into().expect("three rows"),
                parts: parts
                    .into_iter()
                    .map(|row| match row.try_into() {
                        Ok(three) => three,
                        Err(_) => panic!("the plan's three stack rows a row of parts"),
                    })
                    .collect(),
                of_id: (0..n).map(|_| AtomicU32::new(0)).collect(),
                reading: (0..n).map(|_| AtomicU32::new(0)).collect(),
                slots: Vec::new(),
            });
        }
        let count = slots_of(plan.host.nvme_arena_bytes, &slot_sizes)
            .map_err(|e| GpuError::shape(WHAT, e))?;
        let page = page_bytes().map_err(|e| GpuError::plan(WHAT, e))?;
        let bytes = count * slot_sizes.iter().sum::<usize>();
        let map = AnonMap::new(bytes)
            .map_err(|e| GpuError::shape(WHAT, format!("the arena's {bytes} B: {e}")))?;
        let bases = layer_bases(count, &slot_sizes);
        for (arena, base) in arenas.iter_mut().flatten().zip(bases) {
            arena.base = base;
            arena.slots = (0..count)
                .map(|_| SlotBook {
                    state: AtomicU32::new(FREE),
                    id: AtomicU32::new(u32::MAX),
                    tick: AtomicU64::new(0),
                })
                .collect();
        }
        Ok(Some(NvTier {
            shared: SharedArena(map.as_ptr() as *mut u8),
            map,
            by_layer: arenas,
            books: Mutex::new(Books {
                tick: 0,
                seen: Vec::new(),
            }),
            stats: Stats::default(),
            budget: plan.host.nvme_arena_bytes,
            paged: plan.host.nvme_expert_bytes,
            audit: OnceLock::new(),
            split: Arc::clone(split),
            page,
        }))
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

    /// The arena as the leg's [`TierSlots`] handle.
    pub(crate) fn slots(self: &Arc<Self>) -> Arc<dyn TierSlots> {
        self.clone()
    }

    /// Resolve the host leg's part indices — the names of the layer's three
    /// routed stacks as the leg built it, `[gate, up, down]` — to the plan's
    /// stack rows the arena carved, refusing by name a stack the arena does
    /// not hold or a name twice. Load-time only, once; until it runs every
    /// pick of the layer refuses by name.
    pub(crate) fn note_parts(&self, layer: usize, leg: &[&str; PARTS]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::note_parts";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(GpuError::state(WHAT, "the layer's arena"));
        };
        let mut part_of = [usize::MAX; PARTS];
        for (p, name) in leg.iter().enumerate() {
            let Some(at) = arena.names.iter().position(|n| n == name) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {layer}'s stack {name} is not one of the plan's routed rows {:?}",
                        arena.names
                    ),
                ));
            };
            if part_of.contains(&at) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {layer}'s stacks name the plan's row {name} twice"),
                ));
            }
            part_of[p] = at;
        }
        arena
            .part_of
            .set(part_of)
            .map_err(|_| GpuError::state(WHAT, "the layer's part mapping, set twice"))
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
            drops: s.drops.load(Ordering::Relaxed),
            drop_bytes: s.drop_bytes.load(Ordering::Relaxed),
            drop_ns: s.drop_ns.load(Ordering::Relaxed),
        }
    }

    /// The arena's budget in bytes.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// The arena's mapping as an address range `[start, end)`: the only
    /// pages its evictions return.
    #[must_use]
    pub fn range(&self) -> (usize, usize) {
        let start = self.map.as_ptr() as usize;
        (start, start + self.map.bytes().len())
    }

    /// The routed-expert bytes the plan pages on the NVMe tier, which the
    /// arena serves beside the card's victims.
    #[must_use]
    pub fn paged_bytes(&self) -> u64 {
        self.paged
    }

    /// A lane is about to read layer `layer`'s expert `id` through the model
    /// file's mapping: until [`NvTier::end_read`] no drop names a page of it
    /// — the bytes it faults in stay until it has copied them out. Nothing
    /// for an id the arena does not serve. Refused by name: a layer the
    /// arena does not cover and an id past its experts.
    pub fn open_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::open_read";
        let arena = self.arena_of(layer, WHAT)?;
        if arena.host.contains(&id) {
            return Ok(());
        }
        arena.open(layer, id, WHAT)
    }

    /// The lane that opened layer `layer`'s expert `id`
    /// ([`NvTier::open_read`]) has copied it out: the id leaves the mapping
    /// and the page cache, unless another lane still has it open, whose own
    /// end drops it ([`NvTier::drop_layer`]'s rule for the rest). Refused by
    /// name: an end with no open, besides [`NvTier::open_read`]'s refusals
    /// and a drop the kernel refuses.
    pub fn end_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::end_read";
        let arena = self.arena_of(layer, WHAT)?;
        if arena.host.contains(&id) {
            return Ok(());
        }
        arena.close(layer, id, WHAT)?;
        let mut marks = arena.live_marks();
        let mark = &mut marks[id as usize];
        if *mark == Mark::Rest {
            *mark = Mark::Read;
            self.drop_marked(layer, arena, &marks, WHAT)?;
        }
        Ok(())
    }

    /// A union call has read layer `layer`'s experts through the model
    /// file's mapping and consumed them: every id the arena serves
    /// ([`NvTier::serves`]) leaves this process's page tables, then the page
    /// cache ([`PageDrop`]) — the ids the call read, the ones the kernel read
    /// around them, and the ones nobody read, at no cost but the walk —
    /// except the ids a lane has open ([`NvTier::open_read`]), whose own end
    /// drops them. The pages are the whole ones [`drop_runs`] names: never
    /// one that holds a byte of the plan's host segment, of an id a lane has
    /// open, or of the tensors beside a stack. Refused by name: a layer the
    /// arena does not cover, and a drop the kernel refuses.
    pub fn drop_layer(&self, layer: usize) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::drop_layer";
        let arena = self.arena_of(layer, WHAT)?;
        self.drop_marked(layer, arena, &arena.layer_marks(), WHAT)
    }

    /// A prompt call's union over `lists` has read layer `layer`'s experts
    /// through the model file's mapping and consumed them: when it listed an
    /// id the arena serves, the whole layer's go ([`NvTier::drop_layer`]);
    /// one that listed only the host segment's ids read none of them. The
    /// batch port's alone ([`super::HostExperts::release_union`]).
    pub fn release_union(&self, layer: usize, lists: &[&[(u32, f32)]]) -> Result<(), GpuError> {
        if !lists
            .iter()
            .any(|l| l.iter().any(|&(id, _)| self.serves(layer, id)))
        {
            return Ok(());
        }
        self.drop_layer(layer)
    }

    /// The pages [`drop_runs`] names over each stack row of `layer` under
    /// `marks`, out of the mapping and the page cache, and the drop counted;
    /// nothing when no id is read.
    fn drop_marked(
        &self,
        layer: usize,
        arena: &LayerArena,
        marks: &[Mark],
        what: &'static str,
    ) -> Result<(), GpuError> {
        if !marks.contains(&Mark::Read) {
            return Ok(());
        }
        let t0 = Instant::now();
        let mut pages = PageDrop::new(&self.split);
        for (shard, runs) in self.runs_of(arena, marks, what)? {
            for run in runs {
                pages.release(shard, run).map_err(|e| {
                    GpuError::plan(what, format!("layer {layer} shard {shard}: {e}"))
                })?;
            }
        }
        self.stats.drops.fetch_add(1, Ordering::Relaxed);
        self.stats
            .drop_bytes
            .fetch_add(pages.bytes(), Ordering::Relaxed);
        self.stats
            .drop_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
            for (row, (shard, runs)) in self.drop_region(layer)?.into_iter().enumerate() {
                let Some((_, pages)) = held.iter().find(|(f, _)| *f == HostFile::Shard(shard))
                else {
                    continue;
                };
                if let Some(at) = overlap_of(&runs, pages, self.page) {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {layer}: the host set holds bytes {}..{} of {} (expert {}), \
                             which the NVMe tier drops after each read: a churn pool on a paged \
                             stack would lose its pages under the host",
                            at.start,
                            at.end,
                            arena.names[row],
                            arena.id_at(row, at.start)
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Per stack row of `arena`, its shard and the runs [`drop_runs`] names
    /// under `marks`; a refusal of the books names `what` and the row.
    fn runs_of(
        &self,
        arena: &LayerArena,
        marks: &[Mark],
        what: &'static str,
    ) -> Result<DropRuns, GpuError> {
        let mut ranges = Vec::with_capacity(arena.parts.len());
        let mut out = Vec::with_capacity(PARTS);
        for row in 0..PARTS {
            ranges.clear();
            ranges.extend(
                arena
                    .parts
                    .iter()
                    .map(|p| p[row].at..p[row].at + p[row].len as u64),
            );
            let mut runs = Vec::new();
            drop_runs(&ranges, marks, self.page, &mut runs)
                .map_err(|e| GpuError::shape(what, format!("{}: {e}", arena.names[row])))?;
            out.push((arena.shards[row], runs));
        }
        Ok(out)
    }

    /// Fill the slots of `layer`'s `ids` the arena does not already hold:
    /// the misses' parts read with [`DirectFile::read_span`] into their
    /// slots' windows, split over the resident pool, the victims' pages
    /// returned (`MADV_DONTNEED` on the arena's own range only). A read
    /// that fails names its file, offset and errno; a slot whose fill
    /// failed is freed, never lent half-filled. An id the plan's host
    /// segment holds is the mapping's and is skipped ([`NvTier::serves`]).
    pub fn ensure(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        self.fill_missing(layer, ids)?;
        if self.audit.get().is_some() {
            self.audit_slots(layer, ids)?;
        }
        Ok(())
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
            for f in &arena.files {
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
            for part in 0..PARTS {
                let r = arena.parts[id as usize][part];
                let (at, len) = window_span(arena.slot_off(slot), &arena.windows, part, r);
                let held = &self.map.bytes()[at..at + len];
                let path = arena.files[part].path();
                let (_, file) = files.iter().find(|(p, _)| p == path).ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("{} was not opened for the audit", path.display()),
                    )
                })?;
                for off in audit_offsets(r.at, r.len) {
                    let n = DIRECT_ALIGN.min(r.len - off);
                    file.read_exact_at(&mut fresh[..n], r.at + off as u64)
                        .map_err(|e| {
                            GpuError::shape(
                                WHAT,
                                format!("read {} at {}: {e}", path.display(), r.at + off as u64),
                            )
                        })?;
                    if fresh[..n] != held[off..off + n] {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "layer {layer} expert {id} part {part}: the slot's {n} B at +{off} \
                                 are not the file's at {}",
                                r.at + off as u64
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// The misses of `ids` read into their slots ([`NvTier::ensure`]).
    fn fill_missing(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::ensure";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(GpuError::state(WHAT, "the layer's arena"));
        };
        let claim = {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            claim_slots(arena, &mut books, ids).map_err(|e| GpuError::shape(WHAT, e))?
        };
        self.stats.misses.fetch_add(claim.misses, Ordering::Relaxed);
        for &slot in &claim.evicted {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            self.stats
                .resident_bytes
                .fetch_sub(arena.slot_bytes as u64, Ordering::Relaxed);
            self.map
                .dontneed(arena.slot_off(slot), arena.slot_bytes)
                .map_err(|e| GpuError::plan(WHAT, e))?;
        }
        let picks = claim.picks;
        if picks.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        self.fill(arena, &picks).inspect_err(|_| {
            // A failed fill frees its slot: never lent half-filled.
            for p in &picks {
                arena.slots[p.slot].state.store(FREE, Ordering::Relaxed);
                arena.of_id[p.id as usize].store(0, Ordering::Relaxed);
            }
        })?;
        {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            for p in &picks {
                books.tick += 1;
                arena.slots[p.slot]
                    .tick
                    .store(books.tick, Ordering::Relaxed);
                arena.slots[p.slot].state.store(FILLED, Ordering::Relaxed);
            }
        }
        let bytes: u64 = picks
            .iter()
            .map(|p| {
                arena.parts[p.id as usize]
                    .iter()
                    .map(|r| r.len as u64)
                    .sum::<u64>()
            })
            .sum();
        self.stats
            .fills
            .fetch_add(picks.len() as u64, Ordering::Relaxed);
        self.stats.fill_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.stats
            .fill_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stats.resident_bytes.fetch_add(
            picks.len() as u64 * arena.slot_bytes as u64,
            Ordering::Relaxed,
        );
        Ok(())
    }

    /// The misses' reads, split over the resident pool: each worker fills
    /// the picks its chunk names, part by part into its slot's own windows
    /// — disjoint by the books' slot ownership, joined before this returns.
    fn fill(&self, arena: &LayerArena, picks: &[Pick]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::fill";
        let fail: Mutex<Option<GpuError>> = Mutex::new(None);
        threads::pool().for_each_chunk(picks.len(), |chunk| {
            for p in &picks[chunk] {
                for part in 0..PARTS {
                    let r = arena.parts[p.id as usize][part];
                    let (start, span, need) = aligned_span(r.at, r.len as u64);
                    // SAFETY: the pick owns its slot (the books marked it
                    // Filling) and this worker writes only this part's
                    // window of it; the workers' windows are disjoint by
                    // their picks' slots, and `for_each_chunk` joins every
                    // worker before it returns.
                    let window = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.shared
                                .0
                                .add(arena.slot_off(p.slot) + arena.windows[part]),
                            span,
                        )
                    };
                    let got = match arena.files[part].read_span(window, start) {
                        Ok(n) => n,
                        Err(e) => return note(&fail, GpuError::plan(WHAT, e)),
                    };
                    if got < need {
                        return note(
                            &fail,
                            GpuError::shape(
                                WHAT,
                                format!(
                                    "expert {}: the run ends {got} B in, its part needs {need} B \
                                     at {start}",
                                    p.id
                                ),
                            ),
                        );
                    }
                    if !arena.files[part].is_direct() {
                        self.stats.buffered_reads.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .buffered_bytes
                            .fetch_add(r.len as u64, Ordering::Relaxed);
                    }
                }
            }
        });
        fail.into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .map_or(Ok(()), Err)
    }
}

/// Record the first fill failure, the rest dropped on the floor.
fn note(fail: &Mutex<Option<GpuError>>, e: GpuError) {
    let mut cell = fail.lock().unwrap_or_else(|e| e.into_inner());
    cell.get_or_insert(e);
}

impl TierSlots for NvTier {
    fn ensure(&self, layer: usize, ids: &[u32]) -> Result<(), TierError> {
        NvTier::ensure(self, layer, ids).map_err(|e| TierError::Fill {
            layer,
            id: ids.first().copied().unwrap_or(u32::MAX),
            source: Box::new(e),
        })
    }

    fn slot(&self, layer: usize, id: u32, part: usize) -> Option<&[u8]> {
        let arena = self.by_layer.get(layer)?.as_ref()?;
        let part = *arena.part_of.get()?.get(part)?;
        let slot = arena.filled(id)?;
        let r = arena.parts[id as usize][part];
        let (at, len) = window_span(arena.slot_off(slot), &arena.windows, part, r);
        Some(&self.map.bytes()[at..at + len])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FILLED, FILLING, FREE, PARTS, PartRange, SlotBook, advice_disjoint, audit_offsets,
        layer_bases, pick_slot, slots_of, window_bytes, window_span,
    };
    use engram::direct::{DIRECT_ALIGN, aligned_span};
    use std::sync::atomic::{AtomicU32, AtomicU64};

    fn book(state: u32, tick: u64) -> SlotBook {
        SlotBook {
            state: AtomicU32::new(state),
            id: AtomicU32::new(u32::MAX),
            tick: AtomicU64::new(tick),
        }
    }

    /// A part's window holds every aligned span its own byte range can ask
    /// for, at any skew the file can leave it at, and no window crosses a
    /// slot or another part's.
    #[test]
    fn a_parts_window_holds_its_aligned_spans() {
        let a = DIRECT_ALIGN;
        for len in [a / 2, a, 3 * a + 100, 921_600] {
            let w = window_bytes(len);
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

    /// The slots a budget holds: the same count a layer, the carve under
    /// the budget; a budget under one slot of every layer is the named
    /// refusal, and so is a plan that pages nothing.
    #[test]
    fn the_budget_carves_the_same_count_a_layer() {
        let sizes = [1_000_000, 1_200_000, 900_000];
        let total: u64 = sizes.iter().sum::<usize>() as u64;
        let n = slots_of(10 * total, &sizes).expect("ten tiers of slots");
        assert_eq!(n, 10);
        assert!(n as u64 * total <= 10 * total);
        assert_eq!(slots_of(total, &sizes), Ok(1));
        let under = slots_of(total - 1, &sizes).unwrap_err();
        assert!(
            under.contains("under one slot") && under.contains(&(total - 1).to_string()),
            "{under}"
        );
        assert!(
            slots_of(total, &[])
                .unwrap_err()
                .contains("no routed stack")
        );
    }

    /// The slot a miss takes: the first free one, else the least recently
    /// used filled; a filling slot is nobody's to take.
    #[test]
    fn a_miss_takes_a_free_slot_else_the_least_recently_used() {
        let free = [book(FREE, 0), book(FILLED, 5), book(FILLED, 2)];
        assert_eq!(pick_slot(&free), Some(0), "the first free slot");
        let lru = [book(FILLED, 5), book(FILLED, 2), book(FILLED, 9)];
        assert_eq!(pick_slot(&lru), Some(1), "the least recently used");
        let mixed = [book(FILLING, 1), book(FILLED, 7), book(FILLED, 3)];
        assert_eq!(pick_slot(&mixed), Some(2), "a filling slot is not taken");
        assert_eq!(pick_slot(&[book(FILLING, 1)]), None, "all filling: raced");
        assert_eq!(pick_slot(&[]), None, "no slots at all");
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

    /// A one-layer arena of `slots` slots over `n_expert` experts and no host
    /// segment, its three files one scratch file the books never read.
    fn arena_of(n_expert: u32, slots: usize) -> super::LayerArena {
        let path = std::env::temp_dir().join(format!("nvtier-books-{}", std::process::id()));
        std::fs::write(&path, [0u8; 16]).unwrap();
        let file = std::sync::Arc::new(engram::direct::DirectFile::open_buffered(&path).unwrap());
        let part = PartRange {
            at: 0,
            len: 1,
            skew: 0,
        };
        super::LayerArena {
            host: 0..0,
            n_expert,
            windows: [0; PARTS],
            slot_bytes: DIRECT_ALIGN,
            base: 0,
            part_of: std::sync::OnceLock::new(),
            files: [file.clone(), file.clone(), file],
            shards: [0; PARTS],
            names: Default::default(),
            parts: vec![[part; PARTS]; n_expert as usize],
            of_id: (0..n_expert).map(|_| AtomicU32::new(0)).collect(),
            reading: (0..n_expert).map(|_| AtomicU32::new(0)).collect(),
            slots: (0..slots).map(|_| book(FREE, 0)).collect(),
        }
    }

    /// Mark what a call claimed as filled, as the fill's join does.
    fn fill_claimed(arena: &super::LayerArena, books: &mut super::Books, c: &super::Claim) {
        for p in &c.picks {
            books.tick += 1;
            arena.slots[p.slot]
                .tick
                .store(books.tick, std::sync::atomic::Ordering::Relaxed);
            arena.slots[p.slot]
                .state
                .store(FILLED, std::sync::atomic::Ordering::Relaxed);
        }
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
        let mut books = super::Books {
            tick: 0,
            seen: Vec::new(),
        };
        let first = super::claim_slots(&arena, &mut books, &[0, 1]).unwrap();
        assert_eq!((first.picks.len(), first.misses), (2, 2));
        fill_claimed(&arena, &mut books, &first);
        // Expert 2 takes the slot of the least recently used, expert 0.
        let evict = super::claim_slots(&arena, &mut books, &[2]).unwrap();
        assert_eq!(evict.evicted.len(), 1);
        fill_claimed(&arena, &mut books, &evict);
        assert!(arena.filled(0).is_none(), "expert 0 was evicted");
        // The call that needs expert 0 again claims a slot for it.
        let again = super::claim_slots(&arena, &mut books, &[0]).unwrap();
        assert_eq!(again.picks.len(), 1, "an evicted id is refilled");
        fill_claimed(&arena, &mut books, &again);
        assert!(arena.filled(0).is_some(), "its slot reads filled");
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
            windows[p] = windows[p - 1] + window_bytes(lens[p - 1]);
        }
        let slot_bytes = windows[PARTS - 1] + window_bytes(lens[PARTS - 1]);
        assert_eq!(slot_bytes % a, 0, "the slot is aligned");
        for (p, w) in windows.iter().enumerate() {
            assert_eq!(w % a, 0, "part {p}'s window is aligned");
        }
        for slot in [0usize, 1, 7] {
            for p in 0..PARTS {
                let r = PartRange {
                    at: 0,
                    len: lens[p],
                    skew: a - 1,
                };
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
