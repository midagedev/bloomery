//! The host tier held in RAM: the file bytes of the plan's host segments,
//! read and locked where they lie — in the shards' own mappings, nothing
//! copied. One derivation, [`HostSet`], names those pages; a load populates
//! them ([`HostSet::populate`]) so that no decode step takes the first-touch
//! read, and a serving process may also lock them ([`HostLock`]) so that no
//! other reader of the page cache can evict an expert page from under a step.
//! Both walk the same set, so the populated bytes and the locked bytes are
//! the same bytes by construction. A host tier that reads its routed gates
//! and ups from the r8 sidecar has a set of both files ([`HostSet::of_r8`]):
//! those stacks' runs in the sidecar's mapping, everything else in the
//! shards'.
//!
//! Every walk over a set — the populate, the lock and the residency count —
//! cuts the set's files into the same chunks of about equal pages, a thread
//! each: one thread reads a file's pages one fault's read-around at a time,
//! and the device serves several at once, so no file, however large, is left
//! to one thread.
//!
//! The complement lives here too: [`PageDrop`] releases file bytes the load
//! has put on a card and no later reader needs, so that the upload does not
//! leave them in the page cache in place of the host tier's.

use std::fmt;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use gguf::{Gguf, Split};

use super::{Device, Format, ModelTensor, PlacementError, Plan};
use crate::r8file::Sidecar;

/// Bytes per page of this host (`sysconf(_SC_PAGESIZE)`), the unit a lock is
/// taken and counted in.
#[must_use]
pub fn page_bytes() -> u64 {
    // SAFETY: `sysconf` reads a static system value and touches no memory of
    // ours; a negative return means the name is unknown, which _SC_PAGESIZE
    // never is on the hosts this crate builds for.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).map_or(4096, |p| p.max(4096))
}

/// About how many chunks, a thread each, a walk cuts a set into: several
/// times the reads one thread keeps in flight, so that the device, not one
/// thread, bounds the walk.
const WALK_CHUNKS: u64 = 16;

/// A file of a host set: a shard of the split, or the r8 sidecar, which
/// messages name by its path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostFile {
    Shard(usize),
    Sidecar(PathBuf),
}

impl fmt::Display for HostFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostFile::Shard(s) => write!(f, "shard {s}"),
            HostFile::Sidecar(p) => write!(f, "sidecar {}", p.display()),
        }
    }
}

/// The pages of every host segment a plan keeps, per shard: page ranges
/// `[first, end)` of the shard's mapping, sorted and merged where they touch;
/// and, for a host tier that reads the r8 sidecar, the same of its mapping.
#[derive(Clone, Debug)]
pub struct HostSet {
    page: u64,
    /// Only shards with at least one range, in shard order.
    shards: Vec<(usize, Vec<Range<u64>>)>,
    /// The sidecar and its ranges ([`HostSet::of_r8`]).
    side: Option<(Arc<Sidecar>, Vec<Range<u64>>)>,
}

/// One chunk of a walk over a [`HostSet`]: one thread's pages of one file.
#[derive(Clone, Debug)]
pub struct ChunkWalk {
    pub file: HostFile,
    /// Page spans walked: the host segments' ranges, merged where they
    /// touch, cut where two chunks meet.
    pub spans: usize,
    /// Bytes walked, whole pages.
    pub bytes: u64,
    /// This chunk's calls end to end — the reads of the pages that were not
    /// in the page cache.
    pub wall: Duration,
}

/// One file's chunks of a [`Walk`].
#[derive(Clone, Debug)]
pub struct FileWalk {
    pub file: HostFile,
    /// Its chunks, a thread each.
    pub chunks: usize,
    /// Its chunks' spans.
    pub spans: usize,
    /// Its chunks' bytes, whole pages.
    pub bytes: u64,
    /// Its slowest chunk's wall.
    pub wall: Duration,
}

/// A walk over every file of a [`HostSet`], one thread per chunk.
#[derive(Clone, Debug)]
pub struct Walk {
    chunks: Vec<ChunkWalk>,
    wall: Duration,
}

impl Walk {
    /// Bytes walked, whole pages, over every shard and the sidecar.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.bytes).sum()
    }

    /// The part of [`Walk::bytes`] in the sidecar's mapping; 0 without one.
    #[must_use]
    pub fn sidecar_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .filter(|c| matches!(c.file, HostFile::Sidecar(_)))
            .map(|c| c.bytes)
            .sum()
    }

    /// Per chunk, in file order and in page order within a file.
    #[must_use]
    pub fn chunks(&self) -> &[ChunkWalk] {
        &self.chunks
    }

    /// Per file, in file order.
    #[must_use]
    pub fn files(&self) -> Vec<FileWalk> {
        let mut out: Vec<FileWalk> = Vec::new();
        for c in &self.chunks {
            match out.last_mut() {
                Some(f) if f.file == c.file => {
                    f.chunks += 1;
                    f.spans += c.spans;
                    f.bytes += c.bytes;
                    f.wall = f.wall.max(c.wall);
                }
                _ => out.push(FileWalk {
                    file: c.file.clone(),
                    chunks: 1,
                    spans: c.spans,
                    bytes: c.bytes,
                    wall: c.wall,
                }),
            }
        }
        out
    }

    /// The whole walk end to end, all chunks' threads.
    #[must_use]
    pub fn wall(&self) -> Duration {
        self.wall
    }
}

/// One file's pages in a [`HostSet`] and how many of them `mincore` found
/// resident ([`HostSet::resident`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileResidency {
    pub file: HostFile,
    pub resident: u64,
    pub pages: u64,
}

impl HostSet {
    /// The file bytes of every segment `plan` puts on the host in the file's
    /// format whose tensor `keep` selects: each run of consecutive experts of
    /// an expert stack's list, or the whole tensor. Each run grows outward to
    /// whole pages and the pages merge per shard. `plan` must be a plan of
    /// `split`'s tensors.
    pub fn of(
        split: &Split,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
    ) -> Result<HostSet, PlacementError> {
        HostSet::build(split, None, plan, keep)
    }

    /// [`HostSet::of`] for a host tier that reads every tensor `sidecar`
    /// holds from it: those tensors' runs are pages of the sidecar's mapping,
    /// at its offsets — the sidecar keeps each expert's byte range of the
    /// stack — and pages of no shard; every other tensor's are the shards'.
    pub fn of_r8(
        split: &Split,
        sidecar: &Arc<Sidecar>,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
    ) -> Result<HostSet, PlacementError> {
        HostSet::build(split, Some(sidecar), plan, keep)
    }

    fn build(
        split: &Split,
        sidecar: Option<&Arc<Sidecar>>,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
    ) -> Result<HostSet, PlacementError> {
        let page = page_bytes();
        let (pages, side) = host_pages(split, sidecar.map(Arc::as_ref), plan, keep, page)?;
        let shards = pages
            .into_iter()
            .enumerate()
            .filter(|(_, runs)| !runs.is_empty())
            .collect();
        Ok(HostSet {
            page,
            shards,
            side: sidecar.map(|s| (Arc::clone(s), side)),
        })
    }

    /// Bytes of the set, whole pages.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pages() * self.page
    }

    /// Pages of the set, the sidecar's included.
    #[must_use]
    pub fn pages(&self) -> u64 {
        self.shards
            .iter()
            .flat_map(|(_, runs)| runs)
            .chain(self.side.iter().flat_map(|(_, runs)| runs))
            .map(|r| r.end - r.start)
            .sum()
    }

    /// Pages of the set per file, in file order: each shard with ranges,
    /// then the sidecar, when the set reads one.
    #[must_use]
    pub fn files(&self) -> Vec<(HostFile, u64)> {
        let pages = |runs: &[Range<u64>]| runs.iter().map(|r| r.end - r.start).sum();
        let mut out: Vec<(HostFile, u64)> = self
            .shards
            .iter()
            .map(|(s, runs)| (HostFile::Shard(*s), pages(runs)))
            .collect();
        if let Some((side, runs)) = &self.side {
            out.push((HostFile::Sidecar(side.path().to_path_buf()), pages(runs)));
        }
        out
    }

    /// The sidecar whose pages the set holds, when it holds some.
    #[must_use]
    pub(crate) fn sidecar(&self) -> Option<&Arc<Sidecar>> {
        self.side.as_ref().map(|(s, _)| s)
    }

    /// Read every page of the set into the page cache and map it in `split`'s
    /// mappings (`MADV_POPULATE_READ`), one thread per chunk. A page already
    /// cached costs its page-table entry only.
    pub fn populate(&self, split: &Split) -> Result<Walk, PlacementError> {
        walk(split, self, |span| {
            // SAFETY: `span` is a live sub-slice of a read-only file mapping;
            // MADV_POPULATE_READ reads its pages in and maps them, and never
            // writes them.
            let rc = unsafe {
                libc::madvise(
                    span.as_ptr().cast_mut().cast(),
                    span.len(),
                    libc::MADV_POPULATE_READ,
                )
            };
            if rc == 0 {
                Ok(0)
            } else {
                Err(format!(
                    "madvise(MADV_POPULATE_READ): {}",
                    io::Error::last_os_error()
                ))
            }
        })
        .1
        .map(|(w, _)| w)
    }

    /// Per file, the pages of the set `mincore` reports resident in the page
    /// cache, and all of them, over the same chunks as a populate, a thread
    /// each. `split` must be the split the set was made from.
    pub fn resident(&self, split: &Split) -> Result<Vec<FileResidency>, PlacementError> {
        let page = self.page;
        let (walked, counts) = walk(split, self, |span| {
            resident_pages(span, page)
                .map(|(n_in, _)| n_in)
                .map_err(|e| format!("mincore: {e}"))
        })
        .1?;
        let mut out: Vec<FileResidency> = Vec::new();
        for (c, n_in) in walked.chunks.iter().zip(counts) {
            let n = c.bytes / page;
            match out.last_mut() {
                Some(f) if f.file == c.file => {
                    f.resident += n_in;
                    f.pages += n;
                }
                _ => out.push(FileResidency {
                    file: c.file.clone(),
                    resident: n_in,
                    pages: n,
                }),
            }
        }
        Ok(out)
    }
}

/// Page ranges `[first, end)` of one mapping.
type Runs = Vec<Range<u64>>;

/// One file of a set: its name, its mapping and its ranges.
type FileRuns<'a> = (HostFile, &'a Gguf, &'a [Range<u64>]);

/// The files of `set` in file order — each shard with ranges, then the
/// sidecar — with their mappings and ranges.
fn file_runs<'a>(split: &'a Split, set: &'a HostSet) -> Result<Vec<FileRuns<'a>>, PlacementError> {
    let mut out = Vec::with_capacity(set.shards.len() + 1);
    for (s, runs) in &set.shards {
        out.push((HostFile::Shard(*s), shard_of(split, *s)?, runs.as_slice()));
    }
    if let Some((side, runs)) = &set.side {
        let file = HostFile::Sidecar(side.path().to_path_buf());
        out.push((file, side.gguf(), runs.as_slice()));
    }
    Ok(out)
}

/// One thread's part of a walk: a file, its mapping, and page ranges of it
/// in order.
struct Chunk<'a> {
    file: HostFile,
    g: &'a Gguf,
    runs: Runs,
}

/// `set`'s files cut into chunks, in file order and in page order within a
/// file: each file's pages into `⌈pages / per⌉` chunks ([`cut`]), `per`
/// being the set's pages over [`WALK_CHUNKS`]. The set alone fixes the cut.
fn chunks<'a>(split: &'a Split, set: &'a HostSet) -> Result<Vec<Chunk<'a>>, PlacementError> {
    let per = set.pages().div_ceil(WALK_CHUNKS).max(1);
    let mut out = Vec::new();
    for (file, g, runs) in file_runs(split, set)? {
        let pages: u64 = runs.iter().map(|r| r.end - r.start).sum();
        for piece in cut(runs, pages.div_ceil(per)) {
            out.push(Chunk {
                file: file.clone(),
                g,
                runs: piece,
            });
        }
    }
    Ok(out)
}

/// `runs`, page ranges in order, cut into `n` pieces of equal pages — the
/// first `pages % n` a page more — in order: a run is cut at the page where
/// two pieces meet. `n` is taken between 1 and the pages; no pages, no
/// pieces.
fn cut(runs: &[Range<u64>], n: u64) -> Vec<Runs> {
    let pages: u64 = runs.iter().map(|r| r.end - r.start).sum();
    let n = n.clamp(1, pages.max(1));
    let mut sizes = (0..n).map(|c| pages / n + u64::from(c < pages % n));
    let mut want = sizes.next().unwrap_or(0);
    let mut out = Vec::new();
    let mut piece = Vec::new();
    for r in runs {
        let mut at = r.start;
        while at < r.end {
            let take = (r.end - at).min(want);
            piece.push(at..at + take);
            at += take;
            want -= take;
            if want == 0 {
                out.push(std::mem::take(&mut piece));
                want = sizes.next().unwrap_or(u64::MAX);
            }
        }
    }
    if !piece.is_empty() {
        out.push(piece);
    }
    out
}

/// The pages of a [`HostSet`], locked in the shards' mappings (and the
/// sidecar's) until this drops. Owned: it holds the spans' addresses, not a
/// borrow of the split, so the owner of the split can hold it too; it holds
/// the sidecar, whose mapping its drop unlocks.
///
/// The split's mappings must outlive the lock (drop the lock first). If they
/// go first, the unmapping has already released the pages and the drop's
/// `munlock` over the old addresses changes only residency — never memory
/// contents.
pub struct HostLock {
    /// `(address, length)` of every span locked.
    spans: Vec<(usize, usize)>,
    walk: Walk,
    /// Dropped after the spans are unlocked (fields drop after `drop`).
    _sidecar: Option<Arc<Sidecar>>,
}

impl HostLock {
    /// Lock every page of `set` in `split`'s mappings, one thread per chunk.
    /// A page not yet in the page cache is read first, so a set populated
    /// just before locks at page-table speed. A refusal — `RLIMIT_MEMLOCK`,
    /// most often — unlocks what was taken and names the limit.
    pub fn lock(split: &Split, set: &HostSet) -> Result<HostLock, PlacementError> {
        let (spans, walked) = walk(split, set, |span| {
            // SAFETY: `span` is a live sub-slice of a read-only file mapping;
            // mlock faults its pages in and pins them, and never writes them.
            if unsafe { libc::mlock(span.as_ptr().cast(), span.len()) } == 0 {
                Ok(0)
            } else {
                let e = io::Error::last_os_error();
                Err(format!("mlock: {e} ({})", memlock_limit()))
            }
        });
        let mut lock = HostLock {
            spans,
            walk: Walk {
                chunks: Vec::new(),
                wall: Duration::ZERO,
            },
            _sidecar: set.sidecar().cloned(),
        };
        // An error drops the partial lock, which unlocks what was taken.
        lock.walk = walked?.0;
        Ok(lock)
    }

    /// Bytes locked, whole pages, over every shard and the sidecar.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.walk.bytes()
    }

    /// The part of [`HostLock::bytes`] in the sidecar's mapping.
    #[must_use]
    pub fn sidecar_bytes(&self) -> u64 {
        self.walk.sidecar_bytes()
    }

    /// Per chunk, in file order.
    #[must_use]
    pub fn chunks(&self) -> &[ChunkWalk] {
        self.walk.chunks()
    }

    /// Per file, in file order.
    #[must_use]
    pub fn files(&self) -> Vec<FileWalk> {
        self.walk.files()
    }

    /// The whole lock end to end, all chunks' threads.
    #[must_use]
    pub fn wall(&self) -> Duration {
        self.walk.wall()
    }
}

impl Drop for HostLock {
    fn drop(&mut self) {
        for &(addr, len) in &self.spans {
            // SAFETY: munlock reads no memory through `addr`; it clears the
            // lock bit of the pages mapped there, which are this lock's spans
            // while the split lives (the type's contract). A failure leaves
            // them locked until the mapping goes, and a drop has no one to
            // report it to.
            unsafe {
                libc::munlock(addr as *const libc::c_void, len);
            }
        }
    }
}

/// `RLIMIT_MEMLOCK` of this process, for an `mlock` refusal's message.
fn memlock_limit() -> String {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes one `rlimit` into `lim`, which lives here.
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &raw mut lim) } != 0 {
        return "RLIMIT_MEMLOCK unreadable; see `ulimit -l`".into();
    }
    let show = |v: libc::rlim_t| {
        if v == libc::RLIM_INFINITY {
            "unlimited".to_string()
        } else {
            format!("{v} B")
        }
    };
    format!(
        "RLIMIT_MEMLOCK soft {} hard {}; raise `ulimit -l` or run with CAP_IPC_LOCK",
        show(lim.rlim_cur),
        show(lim.rlim_max)
    )
}

/// Releases file bytes of a split from this process's mappings and from the
/// page cache: whole pages inside each range only, so a page that also holds
/// bytes outside it stays. For bytes already on a card that no later reader
/// takes from the file.
pub struct PageDrop<'s> {
    split: &'s Split,
    page: u64,
    /// Each shard's descriptor, opened at its first release.
    files: Vec<Option<File>>,
    bytes: u64,
}

impl<'s> PageDrop<'s> {
    #[must_use]
    pub fn new(split: &'s Split) -> PageDrop<'s> {
        PageDrop {
            split,
            page: page_bytes(),
            files: (0..split.shard_count()).map(|_| None).collect(),
            bytes: 0,
        }
    }

    /// Release the whole pages inside bytes `at` of shard `shard`'s mapping
    /// (file offsets). `posix_fadvise(DONTNEED)` skips a folio that is still
    /// mapped, so this process's page-table entries go first
    /// (`MADV_DONTNEED`); a page another process maps stays cached.
    pub fn release(&mut self, shard: usize, at: Range<u64>) -> Result<(), PlacementError> {
        let g = shard_of(self.split, shard)?;
        let page = self.page;
        let file = self.file(shard)?;
        let dropped = drop_pages(&HostFile::Shard(shard), g.mapping(), file, page, at)?;
        self.bytes += dropped;
        Ok(())
    }

    /// Release every whole page of the r8 sidecar `side`, as
    /// [`PageDrop::release`] does a shard's. A resident sidecar is refused by
    /// name: its bytes are an anonymous copy, which `MADV_DONTNEED` would
    /// zero.
    pub fn release_sidecar(&mut self, side: &Sidecar) -> Result<(), PlacementError> {
        let name = HostFile::Sidecar(side.path().to_path_buf());
        if !side.is_mapped() {
            return Err(PlacementError::Host(format!(
                "{name} is a resident copy, not its file's mapping: it has no file pages to release"
            )));
        }
        let file = File::open(side.path())
            .map_err(|e| PlacementError::Host(format!("open {name} to release pages: {e}")))?;
        let map = side.gguf().mapping();
        self.bytes += drop_pages(&name, map, &file, self.page, 0..map.len() as u64)?;
        Ok(())
    }

    /// Bytes released so far, whole pages.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn file(&mut self, shard: usize) -> Result<&File, PlacementError> {
        let path = self
            .split
            .shard_path(shard)
            .ok_or_else(|| PlacementError::Host(format!("shard {shard} is not in the split")))?;
        let slot = self
            .files
            .get_mut(shard)
            .ok_or_else(|| PlacementError::Host(format!("shard {shard} is not in the split")))?;
        if slot.is_none() {
            let f = File::open(path).map_err(|e| {
                PlacementError::Host(format!("open {} to release pages: {e}", path.display()))
            })?;
            *slot = Some(f);
        }
        slot.as_ref()
            .ok_or_else(|| PlacementError::Host(format!("shard {shard}'s file did not open")))
    }
}

/// Drop the whole pages inside bytes `at` of `name`'s mapping `map`, whose
/// file `file` is, from this process's page tables and then from the page
/// cache; the bytes dropped.
fn drop_pages(
    name: &HostFile,
    map: &[u8],
    file: &File,
    page: u64,
    at: Range<u64>,
) -> Result<u64, PlacementError> {
    let first = at.start.div_ceil(page) * page;
    let end = at.end / page * page;
    if end <= first {
        return Ok(0);
    }
    let span = usize::try_from(first)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(a, b)| map.get(a..b))
        .ok_or_else(|| {
            PlacementError::Host(format!(
                "{name} bytes {at:?} run past its {} bytes",
                map.len()
            ))
        })?;
    // SAFETY: `span` is a live sub-slice of a read-only MAP_SHARED file
    // mapping (a split's shards always are; `release_sidecar` refuses a
    // resident sidecar). MADV_DONTNEED on it drops this process's page-table
    // entries only; the next touch re-faults the same file bytes, so no
    // borrow of the mapping ever sees other contents.
    if unsafe {
        libc::madvise(
            span.as_ptr().cast_mut().cast(),
            span.len(),
            libc::MADV_DONTNEED,
        )
    } != 0
    {
        return Err(PlacementError::Host(format!(
            "madvise(MADV_DONTNEED) of {name} bytes {first}..{end}: {}",
            io::Error::last_os_error()
        )));
    }
    let off = |v: u64| {
        libc::off_t::try_from(v)
            .map_err(|_| PlacementError::Host(format!("{name} offset {v} passes off_t")))
    };
    // SAFETY: the descriptor is this file's own and lives across the call;
    // fadvise reads no memory of ours.
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            off(first)?,
            off(end - first)?,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    if rc != 0 {
        return Err(PlacementError::Host(format!(
            "posix_fadvise(DONTNEED) of {name} bytes {first}..{end}: {}",
            io::Error::from_raw_os_error(rc)
        )));
    }
    Ok(end - first)
}

/// What one chunk's thread hands back: every span its call succeeded on —
/// also when a later one failed, so that a lock's drop unlocks them — and
/// its record with the sum of `op`'s counts.
type ChunkResult = (
    Vec<(usize, usize)>,
    Result<(ChunkWalk, u64), PlacementError>,
);

/// A walk and each chunk's count, in chunk order.
type Walked = (Walk, Vec<u64>);

/// Run `op` over every page run of `set` in `split`'s mappings and the
/// sidecar's, one thread per chunk ([`chunks`]), a chunk's runs in order.
/// `op` returns a count each chunk sums (the resident pages of a residency
/// count; 0 for a populate or a lock). Returns the spans `op` succeeded on
/// and the walk, or the first refusal in chunk order.
fn walk(
    split: &Split,
    set: &HostSet,
    op: impl Fn(&[u8]) -> Result<u64, String> + Sync,
) -> (Vec<(usize, usize)>, Result<Walked, PlacementError>) {
    let work = match chunks(split, set) {
        Ok(work) => work,
        Err(e) => return (Vec::new(), Err(e)),
    };
    let start = Instant::now();
    let op = &op;
    let results: Vec<ChunkResult> = thread::scope(|scope| {
        let handles: Vec<_> = work
            .iter()
            .map(|c| (c, scope.spawn(move || walk_chunk(c, set.page, op))))
            .collect();
        handles
            .into_iter()
            .map(|(c, h)| {
                h.join().unwrap_or_else(|_| {
                    let e = PlacementError::Host(format!("{}: a walk thread panicked", c.file));
                    (Vec::new(), Err(e))
                })
            })
            .collect()
    });
    let wall = start.elapsed();
    let mut done = Vec::new();
    let mut walked = Vec::with_capacity(results.len());
    let mut counts = Vec::with_capacity(results.len());
    let mut refused = None;
    for (spans, chunk) in results {
        done.extend(spans);
        match chunk {
            Ok((c, n)) => {
                walked.push(c);
                counts.push(n);
            }
            Err(e) => refused = refused.or(Some(e)),
        }
    }
    match refused {
        Some(e) => (done, Err(e)),
        None => (
            done,
            Ok((
                Walk {
                    chunks: walked,
                    wall,
                },
                counts,
            )),
        ),
    }
}

/// `op` over chunk `c`'s runs, in order; a refusal names the file and the
/// run's bytes.
fn walk_chunk(
    c: &Chunk<'_>,
    page: u64,
    op: &(impl Fn(&[u8]) -> Result<u64, String> + Sync),
) -> ChunkResult {
    let start = Instant::now();
    let mut done = Vec::with_capacity(c.runs.len());
    let (mut bytes, mut count) = (0, 0);
    for r in &c.runs {
        let span = match run_span(c.g, &c.file, r, page) {
            Ok(span) => span,
            Err(e) => return (done, Err(e)),
        };
        match op(span) {
            Ok(n) => count += n,
            Err(e) => {
                let at = r.start * page;
                let e = PlacementError::Host(format!(
                    "{} bytes {at}..{}: {e}",
                    c.file,
                    at + span.len() as u64
                ));
                return (done, Err(e));
            }
        }
        done.push((span.as_ptr() as usize, span.len()));
        bytes += (r.end - r.start) * page;
    }
    let record = ChunkWalk {
        file: c.file.clone(),
        spans: c.runs.len(),
        bytes,
        wall: start.elapsed(),
    };
    (done, Ok((record, count)))
}

fn shard_of(split: &Split, s: usize) -> Result<&Gguf, PlacementError> {
    split
        .shard(s)
        .ok_or_else(|| PlacementError::Host(format!("shard {s} is not in the split")))
}

/// Pages `r` of `g`'s mapping as a slice, the last one clamped to the
/// mapping's end; `file` names the mapping in a refusal.
fn run_span<'g>(
    g: &'g Gguf,
    file: &HostFile,
    r: &Range<u64>,
    page: u64,
) -> Result<&'g [u8], PlacementError> {
    let map = g.mapping();
    let first = r.start * page;
    let end = (r.end * page).min(map.len() as u64);
    usize::try_from(first)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(a, b)| map.get(a..b))
        .ok_or_else(|| {
            PlacementError::Host(format!(
                "{file} bytes {first}..{end} run past its {} bytes",
                map.len()
            ))
        })
}

/// Per shard, the page ranges `[first, end)` of the host segments `keep`
/// selects, sorted and merged where they overlap or touch; with a `sidecar`,
/// the ranges of the tensors it holds go to the second list, in its mapping.
fn host_pages(
    split: &Split,
    sidecar: Option<&Sidecar>,
    plan: &Plan<'_>,
    keep: impl Fn(&ModelTensor) -> bool,
    page: u64,
) -> Result<(Vec<Runs>, Runs), PlacementError> {
    let mut pages: Vec<Runs> = vec![Vec::new(); split.shard_count()];
    let mut side: Runs = Vec::new();
    for row in &plan.rows {
        let Some(t) = plan.model.tensors.get(row.tensor) else {
            return Err(PlacementError::Host(format!(
                "plan row {} names no tensor",
                row.tensor
            )));
        };
        if !keep(t) {
            continue;
        }
        for seg in &row.segments {
            if seg.device != Device::Host || seg.format != Format::HostFile {
                continue;
            }
            let located = split
                .find(&t.name)
                .and_then(|(s, info)| Some((s, info, split.shard(s)?)));
            let Some((s, info, g)) = located else {
                return Err(PlacementError::tensor(
                    t,
                    "is not in the split the host set maps",
                ));
            };
            let held = sidecar.and_then(|sc| sc.find(&t.name).map(|i| (sc.gguf(), i)));
            for span in seg.spans(t, plan.model.experts)? {
                if s != t.shard || span.bytes.end > info.nbytes {
                    return Err(PlacementError::tensor(
                        t,
                        format!(
                            "the plan's shard {} and bytes {:?} are not the split's shard {s} and {} bytes",
                            t.shard, span.bytes, info.nbytes
                        ),
                    ));
                }
                let (base, into) = match held {
                    Some((sg, side_info)) => {
                        if span.bytes.end > side_info.nbytes {
                            return Err(PlacementError::tensor(
                                t,
                                format!(
                                    "the plan's bytes {:?} run past the sidecar's {} bytes of it",
                                    span.bytes, side_info.nbytes
                                ),
                            ));
                        }
                        (sg.data_base() + side_info.offset, &mut side)
                    }
                    None => (g.data_base() + info.offset, &mut pages[s]),
                };
                let (a, b) = (base + span.bytes.start, base + span.bytes.end);
                into.push(a / page..b.div_ceil(page));
            }
        }
    }
    for spans in pages.iter_mut().chain(std::iter::once(&mut side)) {
        spans.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(spans.len());
        for r in spans.drain(..) {
            match merged.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => merged.push(r),
            }
        }
        *spans = merged;
    }
    Ok((pages, side))
}

/// `mincore` over `span`: its resident pages and all of them.
fn resident_pages(span: &[u8], page: u64) -> io::Result<(u64, u64)> {
    let n = (span.len() as u64).div_ceil(page);
    let mut vec = vec![0u8; usize::try_from(n).map_err(io::Error::other)?];
    // SAFETY: `span` starts on a page boundary inside one live mapping (a set
    // run starts at a page multiple of the mapping's page-aligned start), and
    // `vec` has one byte for each of its pages. mincore only reads the page
    // tables; nothing is written through `addr`.
    let rc = unsafe {
        libc::mincore(
            span.as_ptr().cast_mut().cast(),
            span.len(),
            vec.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((vec.iter().filter(|&&b| b & 1 != 0).count() as u64, n))
}

#[cfg(test)]
mod tests {
    use super::cut;
    use std::ops::Range;

    /// A cut keeps every page once and in order, and its pieces differ by at
    /// most a page, into any count from one to the pages: over one run, runs
    /// shorter and longer than a piece (so cuts fall inside runs and on
    /// their edges), and a single page.
    #[test]
    fn a_cut_keeps_every_page_once_and_evens_its_pieces() {
        let shapes: [&[(u64, u64)]; 3] = [
            &[(0, 100)],
            &[(0, 3), (5, 9), (20, 21), (30, 60)],
            &[(7, 8)],
        ];
        for shape in shapes {
            let runs: Vec<Range<u64>> = shape.iter().map(|&(a, b)| a..b).collect();
            let runs = runs.as_slice();
            let want: Vec<u64> = runs.iter().flat_map(Clone::clone).collect();
            let pages = want.len() as u64;
            for n in 1..=pages {
                let pieces = cut(runs, n);
                assert_eq!(pieces.len() as u64, n, "{runs:?} into {n}");
                let sizes: Vec<u64> = pieces
                    .iter()
                    .map(|p| p.iter().map(|r| r.end - r.start).sum())
                    .collect();
                let (lo, hi) = (sizes.iter().min(), sizes.iter().max());
                assert!(
                    lo.zip(hi).is_some_and(|(lo, hi)| *lo >= 1 && hi - lo <= 1),
                    "{runs:?} into {n}: sizes {sizes:?}"
                );
                let got: Vec<u64> = pieces.iter().flatten().flat_map(Clone::clone).collect();
                assert_eq!(got, want, "{runs:?} into {n}");
            }
        }
        assert!(cut(&[], 3).is_empty(), "no pages, no pieces");
    }
}
