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
//! The arena never advises the model mapping: the only pages it returns are
//! its own slots', `MADV_DONTNEED` on an evicted slot's windows, which
//! frees an anonymous range's pages — that is why the arena is anonymous
//! and not the page cache, whose warm set nothing else can flush. The books
//! are one atomic hint an id and a per-slot state word a pick reads without
//! the lock; the LRU and the fills' slot ownership live under one lock
//! taken per `ensure`, never per pick. The engine serves a model's host leg
//! one service at a time, so no pick reads a slot another service evicts —
//! the seat and the run-ahead that share the arena across threads join
//! their fills before the books give the slot away.

use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use engram::direct::{AnonMap, DIRECT_ALIGN, DirectFile, aligned_span};
use gguf::Split;
use model::moe::{TierError, TierSlots};
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

/// Where part `part` of the expert slot `slot` holds `r`'s bytes: the
/// window's base in the mapping and the logical bytes' offset into it.
fn window_span(
    slot_bytes: usize,
    windows: &[usize; PARTS],
    slot: usize,
    part: usize,
    r: PartRange,
) -> (usize, usize) {
    let base = slot * slot_bytes + windows[part];
    (base + r.skew, r.len)
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
    /// The plan's stack row each of the leg's parts reads, resolved at the
    /// attach by name ([`NvTier::note_parts`]); unread until then, and
    /// every pick of the layer refused by name after.
    part_of: OnceLock<[usize; PARTS]>,
    /// The file each of the plan's stack rows reads.
    files: [Arc<DirectFile>; PARTS],
    /// The plan's stack-row tensors' names, for the attach's mapping.
    names: [String; PARTS],
    /// Per id: its parts' ranges, by stack row.
    parts: Vec<[PartRange; PARTS]>,
    /// Per id: the slot that holds it, + 1; 0 when none does.
    of_id: Vec<AtomicU32>,
    /// Per slot: its state (a pick's read), its id and LRU tick (the
    /// lock's alone).
    slots: Vec<SlotBook>,
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
    /// The mapping's first byte, for the pool's fill workers — each fill
    /// writes its own slot's own window, disjoint by the books' slot
    /// ownership.
    shared: SharedArena,
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
    /// a stack whose experts do not cut its bytes evenly, and a budget
    /// under one slot.
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
            for (p, name) in rows.iter().enumerate() {
                let (shard, info) = split
                    .find(name)
                    .ok_or(GpuError::shape(WHAT, format!("{name}: not in the split")))?;
                let gguf = split.shard(shard).ok_or(GpuError::shape(
                    WHAT,
                    format!("{name}: shard {shard} is not there"),
                ))?;
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
                part_of: OnceLock::new(),
                files: rows_files.map(|i| Arc::clone(&files[i])),
                names: rows.clone().try_into().expect("three rows"),
                parts: parts
                    .into_iter()
                    .map(|row| match row.try_into() {
                        Ok(three) => three,
                        Err(_) => panic!("the plan's three stack rows a row of parts"),
                    })
                    .collect(),
                of_id: (0..n).map(|_| AtomicU32::new(0)).collect(),
                slots: Vec::new(),
            });
        }
        let count = slots_of(plan.host.nvme_arena_bytes, &slot_sizes)
            .map_err(|e| GpuError::shape(WHAT, e))?;
        let bytes = count * slot_sizes.iter().sum::<usize>();
        let map = AnonMap::new(bytes)
            .map_err(|e| GpuError::shape(WHAT, format!("the arena's {bytes} B: {e}")))?;
        for arena in arenas.iter_mut().flatten() {
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
        }))
    }

    /// Whether the arena covers `layer` (a paged layer with slots).
    pub(crate) fn covers(&self, layer: usize) -> bool {
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
        }
    }

    /// The arena's budget in bytes.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Fill the slots of `layer`'s `ids` the arena does not already hold:
    /// the misses' parts read with [`DirectFile::read_span`] into their
    /// slots' windows, split over the resident pool, the victims' pages
    /// returned (`MADV_DONTNEED` on the arena's own range only). A read
    /// that fails names its file, offset and errno; a slot whose fill
    /// failed is freed, never lent half-filled.
    pub fn ensure(&self, layer: usize, ids: &[u32]) -> Result<(), GpuError> {
        const WHAT: &str = "NvTier::ensure";
        let Some(arena) = self.by_layer.get(layer).and_then(|l| l.as_ref()) else {
            return Err(GpuError::state(WHAT, "the layer's arena"));
        };
        let picks = {
            let mut books = self.books.lock().unwrap_or_else(|e| e.into_inner());
            if books.seen.len() != arena.n_expert as usize {
                books.seen.clear();
                books.seen.resize(arena.n_expert as usize, false);
            }
            let mut picks = Vec::new();
            for &id in ids {
                let Some(at) = usize::try_from(id)
                    .ok()
                    .filter(|&i| i < arena.n_expert as usize)
                else {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {layer}: expert {id} is past the {} experts",
                            arena.n_expert
                        ),
                    ));
                };
                if std::mem::replace(&mut books.seen[at], true) {
                    continue;
                }
                let held = arena.of_id[at].load(Ordering::Relaxed);
                if held != 0
                    && arena.slots[(held - 1) as usize]
                        .state
                        .load(Ordering::Relaxed)
                        == FILLED
                {
                    books.tick += 1;
                    arena.slots[(held - 1) as usize]
                        .tick
                        .store(books.tick, Ordering::Relaxed);
                    continue;
                }
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                let slot = pick_slot(&arena.slots).ok_or(GpuError::state(
                    WHAT,
                    "every slot is filling: two ensures raced a layer",
                ))?;
                if arena.slots[slot].state.load(Ordering::Relaxed) == FILLED {
                    let gone = arena.slots[slot].id.load(Ordering::Relaxed);
                    arena.of_id[gone as usize].store(0, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .resident_bytes
                        .fetch_sub(arena.slot_bytes as u64, Ordering::Relaxed);
                    self.map
                        .dontneed(slot * arena.slot_bytes, arena.slot_bytes)
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                }
                arena.slots[slot].state.store(FILLING, Ordering::Relaxed);
                arena.slots[slot].id.store(id, Ordering::Relaxed);
                arena.of_id[at].store(slot as u32 + 1, Ordering::Relaxed);
                picks.push(Pick { slot, id });
            }
            picks
        };
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
                                .add(p.slot * arena.slot_bytes + arena.windows[part]),
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
        let slot = arena.of_id.get(id as usize)?.load(Ordering::Relaxed);
        if slot == 0
            || arena.slots[(slot - 1) as usize]
                .state
                .load(Ordering::Relaxed)
                != FILLED
        {
            return None;
        }
        let r = arena.parts[id as usize][part];
        let (at, len) = window_span(arena.slot_bytes, &arena.windows, slot as usize - 1, part, r);
        Some(&self.map.bytes()[at..at + len])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FILLED, FILLING, FREE, PARTS, PartRange, SlotBook, pick_slot, slots_of, window_bytes,
        window_span,
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
                let (at, len) = window_span(slot_bytes, &windows, slot, p, r);
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
