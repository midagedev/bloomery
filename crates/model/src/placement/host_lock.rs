//! The host tier held in RAM: the file bytes of the plan's host segments,
//! read and locked where they lie — in the shards' own mappings, nothing
//! copied. One derivation, [`HostSet`], names those pages; a load populates
//! them ([`HostSet::populate`]) so that no decode step takes the first-touch
//! read, and a serving process may also lock them ([`HostLock`]) so that no
//! other reader of the page cache can evict an expert page from under a step.
//! Both walk the same set, so the populated bytes and the locked bytes are
//! the same bytes by construction. A host tier that reads its routed gates
//! and ups from the r8 sidecar has a set of both files ([`HostSet::of`] over
//! an [`R8Source`] that reads one): those stacks' runs in the sidecar's
//! mapping, everything else in the shards'.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use gguf::{Gguf, Split};
use os::Populated;

use super::{Device, ExpertList, Format, ModelTensor, PlacementError, Plan, Segment};
use crate::ops::{StackRuns, stack_runs};
use crate::r8file::{R8Error, R8Source, Sidecar};

/// Bytes per page of this host (`sysconf(_SC_PAGESIZE)`), the unit a lock is
/// taken and counted in; a host whose base page the row room cannot count in
/// is refused by name.
pub fn page_bytes() -> Result<u64, PlacementError> {
    os::page_bytes().map_err(|e| PlacementError::Host(e.to_string()))
}

/// About how many chunks, a thread each, a walk cuts a set into: several
/// times the reads one thread keeps in flight, so that the device, not one
/// thread, bounds the walk.
pub(crate) const WALK_CHUNKS: u64 = 16;

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
    /// The sidecar the set was built beside and its ranges: every walk's
    /// pair must read this open of it ([`R8Source::reads`]).
    side: Option<(Arc<Sidecar>, Vec<Range<u64>>)>,
    /// What [`HostSet::holds`] answers from when the set also knows runs it
    /// does not walk ([`HostSet::of_known`]): per shard, then the sidecar's,
    /// the walked runs and the known ones merged. `None`: the walked runs.
    held: Option<(Vec<Runs>, Runs)>,
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
    /// `src`'s split's tensors. When `src` reads the r8 sidecar, every tensor
    /// the sidecar holds is read from it: those tensors' runs are pages of
    /// the sidecar's mapping, at its offsets — the sidecar keeps each
    /// expert's byte range of the stack — and pages of no shard; every other
    /// tensor's are the shards'.
    pub fn of(
        src: R8Source<'_>,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
    ) -> Result<HostSet, PlacementError> {
        HostSet::of_with(src, plan, keep, &[])
    }

    /// [`HostSet::of_known`] with no known runs.
    pub fn of_with(
        src: R8Source<'_>,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
        extra: &[(usize, ExpertList)],
    ) -> Result<HostSet, PlacementError> {
        HostSet::of_known(src, plan, keep, extra, &[])
    }

    /// [`HostSet::of`] with `extra` and `known` too: per entry, the experts
    /// of the plan's model tensor at that index which the host must serve
    /// besides its own segments (adaptive residency's churn pool,
    /// [`super::churn`]) — each expert's run in the sidecar when it holds the
    /// tensor, else in the shard. `extra`'s runs are walked as host segments
    /// of theirs are: read in and locked with them, counted in the set's
    /// bytes. `known`'s are held and never walked: [`HostSet::serves`] answers
    /// for them by `mincore`, [`HostSet::populate_run`] reads them in from the
    /// mapping, and the page cache may let them go (a churn pool the host's
    /// room did not take).
    pub fn of_known(
        src: R8Source<'_>,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
        extra: &[(usize, ExpertList)],
        known: &[(usize, ExpertList)],
    ) -> Result<HostSet, PlacementError> {
        let (split, sidecar) = (src.split(), src.sidecar());
        let side_of = sidecar.map(Arc::as_ref);
        let page = page_bytes()?;
        let (pages, side) = host_pages(split, side_of, plan, &keep, extra, page)?;
        let held = if known.is_empty() {
            None
        } else {
            let both: Vec<(usize, ExpertList)> = extra.iter().chain(known).cloned().collect();
            Some(host_pages(split, side_of, plan, &keep, &both, page)?)
        };
        let shards = pages
            .into_iter()
            .enumerate()
            .filter(|(_, runs)| !runs.is_empty())
            .collect();
        Ok(HostSet {
            page,
            shards,
            side: sidecar.map(|s| (Arc::clone(s), side)),
            held,
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
        self.runs()
            .into_iter()
            .map(|(file, runs)| (file, runs.iter().map(|r| r.end - r.start).sum()))
            .collect()
    }

    /// The set's page ranges `[first, end)` per file, in file order: each
    /// shard with ranges, then the sidecar, when the set reads one.
    #[must_use]
    pub fn runs(&self) -> Vec<(HostFile, &[Range<u64>])> {
        let mut out: Vec<(HostFile, &[Range<u64>])> = self
            .shards
            .iter()
            .map(|(s, runs)| (HostFile::Shard(*s), runs.as_slice()))
            .collect();
        if let Some((side, runs)) = &self.side {
            out.push((
                HostFile::Sidecar(side.path().to_path_buf()),
                runs.as_slice(),
            ));
        }
        out
    }

    /// The sidecar whose pages the set holds, when it holds some.
    #[must_use]
    pub(crate) fn sidecar(&self) -> Option<&Arc<Sidecar>> {
        self.side.as_ref().map(|(s, _)| s)
    }

    /// Read every page of the set into the page cache and map it in `src`'s
    /// mappings (`MADV_POPULATE_READ`, else a read a page: [`populate_span`]),
    /// one thread per chunk. A page already cached costs its page-table entry
    /// only. A kernel without `MADV_POPULATE_READ` prints one line.
    pub fn populate(&self, src: R8Source<'_>) -> Result<Walk, PlacementError> {
        let said = AtomicBool::new(false);
        walk(src, self, |span| {
            if populate_span(span, self.page, advise_populate)? == Populated::Touched
                && !said.swap(true, Ordering::Relaxed)
            {
                eprintln!("{NO_POPULATE_READ}");
            }
            Ok(0)
        })
        .1
        .map(|(w, _)| w)
    }

    /// Per file, the pages of the set `mincore` reports resident in the page
    /// cache, and all of them, over the same chunks as a populate, a thread
    /// each. `src`'s split must be the split the set was made from.
    pub fn resident(&self, src: R8Source<'_>) -> Result<Vec<FileResidency>, PlacementError> {
        let page = self.page;
        let (walked, counts) = walk(src, self, |span| {
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

impl HostSet {
    /// Whether every page of bytes `at` of `file`'s mapping is in the set,
    /// walked or known ([`HostSet::of_known`]).
    #[must_use]
    pub fn holds(&self, file: &HostFile, at: &Range<u64>) -> bool {
        if at.end <= at.start {
            return false;
        }
        let pages = at.start / self.page..at.end.div_ceil(self.page);
        let runs: &[Range<u64>] = match (file, &self.held) {
            (HostFile::Shard(s), None) => match self.shards.iter().find(|(x, _)| x == s) {
                Some((_, runs)) => runs,
                None => return false,
            },
            (HostFile::Shard(s), Some((shards, _))) => match shards.get(*s) {
                Some(runs) => runs,
                None => return false,
            },
            (HostFile::Sidecar(p), held) => match &self.side {
                Some((side, runs)) if side.path() == p.as_path() => {
                    held.as_ref().map_or(runs, |(_, known)| known)
                }
                _ => return false,
            },
        };
        let i = runs.partition_point(|r| r.end <= pages.start);
        runs.get(i)
            .is_some_and(|r| r.start <= pages.start && pages.end <= r.end)
    }

    /// Whether bytes `at` of `file`'s mapping in `src` are in the set and
    /// every page of them is in the page cache now (`mincore`): what the host
    /// reads there costs no fault. `src` must read the set's sidecar.
    pub fn serves(
        &self,
        src: R8Source<'_>,
        file: &HostFile,
        at: &Range<u64>,
    ) -> Result<bool, PlacementError> {
        if !self.holds(file, at) {
            return Ok(false);
        }
        let span = self.span(src, file, at)?;
        let (resident, pages) = resident_pages(span, self.page)
            .map_err(|e| PlacementError::Host(format!("mincore of {file} bytes {at:?}: {e}")))?;
        Ok(resident == pages)
    }

    /// Read bytes `at` of `file`'s mapping in `src` into the page cache and
    /// map them (`MADV_POPULATE_READ`), when the set holds them; `false`,
    /// touching nothing, when it does not.
    pub fn populate_run(
        &self,
        src: R8Source<'_>,
        file: &HostFile,
        at: &Range<u64>,
    ) -> Result<bool, PlacementError> {
        if !self.holds(file, at) {
            return Ok(false);
        }
        let span = self.span(src, file, at)?;
        populate_span(span, self.page, advise_populate)
            .map_err(|e| PlacementError::Host(format!("{file} bytes {at:?}: {e}")))?;
        Ok(true)
    }

    /// The whole pages over bytes `at` of `file`'s mapping in `src`.
    fn span<'a>(
        &self,
        src: R8Source<'a>,
        file: &HostFile,
        at: &Range<u64>,
    ) -> Result<&'a [u8], PlacementError> {
        let g = match file {
            HostFile::Shard(s) => shard_of(src.split(), *s)?,
            HostFile::Sidecar(_) => {
                let (side, _) = self.side.as_ref().ok_or_else(|| {
                    PlacementError::Host(format!("{file}: the set reads no sidecar"))
                })?;
                src.reads(side, "a host set's expert run")?;
                src.sidecar()
                    .map(|s| s.gguf())
                    .ok_or_else(|| PlacementError::Host(format!("{file}: the pair reads none")))?
            }
        };
        run_span(
            g,
            file,
            &(at.start / self.page..at.end.div_ceil(self.page)),
            self.page,
        )
    }
}

/// Where the host reads expert `id` of `t`, a routed stack of `experts`:
/// the file and the runs of its mapping — the sidecar's when `src` reads one
/// that holds `t` and `sidecar` asks for it, else the shard's. The expert's
/// one run and an empty second, or for a planes stack a checkpoint folder
/// locates its codes run then its scales run, each where the folder keeps it.
pub fn expert_runs(
    src: R8Source<'_>,
    t: &ModelTensor,
    experts: u64,
    id: u32,
    sidecar: bool,
) -> Result<(HostFile, [Range<u64>; 2]), PlacementError> {
    let (_, per) = super::per_expert(t, experts)?;
    if u64::from(id) >= experts {
        return Err(PlacementError::tensor(
            t,
            format!("expert {id} is not below the stack's {experts}"),
        ));
    }
    let bytes = u64::from(id) * per..(u64::from(id) + 1) * per;
    let split = src.split();
    let (s, info) = split
        .find(&t.name)
        .ok_or_else(|| PlacementError::tensor(t, "is not in the split"))?;
    let held = src
        .sidecar()
        .filter(|_| sidecar)
        .and_then(|sc| sc.find(&t.name).map(|i| (sc, i)));
    Ok(match held {
        Some((sc, i)) => (
            HostFile::Sidecar(sc.path().to_path_buf()),
            [
                sc.gguf().data_base() + i.offset + bytes.start
                    ..sc.gguf().data_base() + i.offset + bytes.end,
                0..0,
            ],
        ),
        None => match runs_of(split, t)? {
            // A stack the folder stores expert by expert: its matrix is where
            // the runs put it, not at an equal stride from the header's
            // offset; a planes matrix is two runs.
            Some(runs) => {
                let ranges = runs.ranges(id as usize).ok_or_else(|| {
                    PlacementError::tensor(t, format!("has no located run for expert {id}"))
                })?;
                (HostFile::Shard(s), ranges)
            }
            None => {
                let base = shard_of(split, s)?.data_base() + info.offset;
                (
                    HostFile::Shard(s),
                    [base + bytes.start..base + bytes.end, 0..0],
                )
            }
        },
    })
}

/// The page ranges of `bytes`, a span of whole experts (`per` bytes each,
/// counted from the stack's first) of a located stack: expert `e`'s matrix is
/// the runs `runs` puts it at, so a span of experts is one range of pages a
/// run, in expert order (a planes expert's codes then its scales), not one
/// range of the span's own length. A span that cuts an expert, or names one
/// past the runs, is the detail of a refusal.
fn located_pages(
    runs: &StackRuns,
    per: u64,
    bytes: &Range<u64>,
    page: u64,
) -> Result<Vec<Range<u64>>, String> {
    let (first, end) = (bytes.start / per, bytes.end / per);
    if !bytes.start.is_multiple_of(per)
        || !bytes.end.is_multiple_of(per)
        || u64::try_from(runs.n_expert()).ok().is_none_or(|n| end > n)
    {
        return Err(format!(
            "the plan's bytes {bytes:?} are not whole experts of the {} located runs",
            runs.n_expert()
        ));
    }
    Ok((first as usize..end as usize)
        .flat_map(|e| runs.ranges(e).expect("an expert below the runs' count"))
        .filter(|r| !r.is_empty())
        .map(|r| r.start / page..r.end.div_ceil(page))
        .collect())
}

/// The runs `split` keeps for stack `t`, when its experts lie at offsets of
/// their own in the shard ([`stack_runs`]).
fn runs_of(split: &Split, t: &ModelTensor) -> Result<Option<StackRuns>, PlacementError> {
    stack_runs(split, &t.name).map_err(|e| PlacementError::tensor(t, e.to_string()))
}

/// Page ranges `[first, end)` of one mapping.
type Runs = Vec<Range<u64>>;

/// One file of a set: its name, its mapping and its ranges.
type FileRuns<'a> = (HostFile, &'a Gguf, &'a [Range<u64>]);

/// The files of `set` in file order — each shard with ranges, then the
/// sidecar — with their mappings and ranges; a set built beside a sidecar is
/// walked only beside a pair that reads that open of it.
fn file_runs<'a>(src: R8Source<'a>, set: &'a HostSet) -> Result<Vec<FileRuns<'a>>, PlacementError> {
    let mut out = Vec::with_capacity(set.shards.len() + 1);
    for (s, runs) in &set.shards {
        out.push((
            HostFile::Shard(*s),
            shard_of(src.split(), *s)?,
            runs.as_slice(),
        ));
    }
    if let Some((side, runs)) = &set.side {
        src.reads(side, "a host set's walk")?;
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
fn chunks<'a>(src: R8Source<'a>, set: &'a HostSet) -> Result<Vec<Chunk<'a>>, PlacementError> {
    let per = set.pages().div_ceil(WALK_CHUNKS).max(1);
    let mut out = Vec::new();
    for (file, g, runs) in file_runs(src, set)? {
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
    /// Lock every page of `set` in `src`'s mappings, one thread per chunk.
    /// A page not yet in the page cache is read first, so a set populated
    /// just before locks at page-table speed. A refusal — `RLIMIT_MEMLOCK`,
    /// most often — unlocks what was taken and names the limit.
    pub fn lock(src: R8Source<'_>, set: &HostSet) -> Result<HostLock, PlacementError> {
        let (spans, walked) = walk(src, set, |span| {
            // SAFETY: `span` is a live sub-slice of one of the set's mappings:
            // a shard's read-only file mapping, or the sidecar's, which is its
            // file's mapping or a resident anonymous copy. mlock faults its
            // pages in and pins them on either kind, and never writes them.
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
            // lock bit of the pages mapped there, which are this lock's spans:
            // a shard's while the split lives (the type's contract), the
            // sidecar's while `_sidecar` holds its mapping — that field drops
            // after this loop. A failure leaves them locked until the mapping
            // goes, and a drop has no one to report it to.
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

/// Releases file bytes of a split — and of the r8 sidecar its pair reads —
/// from this process's mappings and from the page cache: whole pages inside
/// each range only, so a page that also holds bytes outside it stays. For
/// bytes already on a card that no later reader takes from the file.
pub struct PageDrop<'s> {
    src: R8Source<'s>,
    /// Each shard's descriptor, opened at its first release.
    files: Vec<Option<File>>,
    bytes: u64,
}

impl<'s> PageDrop<'s> {
    /// A release of `split`'s shards.
    #[must_use]
    pub fn new(split: &'s Split) -> PageDrop<'s> {
        PageDrop::of(R8Source::rows(split))
    }

    /// A release of `src`'s shards and of the sidecar it reads
    /// ([`PageDrop::release_sidecar`]).
    #[must_use]
    pub fn of(src: R8Source<'s>) -> PageDrop<'s> {
        PageDrop {
            src,
            files: (0..src.split().shard_count()).map(|_| None).collect(),
            bytes: 0,
        }
    }

    /// Release the whole pages inside bytes `at` of shard `shard`'s mapping
    /// (file offsets). `posix_fadvise(DONTNEED)` skips a folio that is still
    /// mapped, so this process's page-table entries go first
    /// (`MADV_DONTNEED`); a page another process maps stays cached.
    pub fn release(&mut self, shard: usize, at: Range<u64>) -> Result<(), PlacementError> {
        let map = FileMapping::shard(self.src.split(), shard)?;
        let file = self.file(shard)?;
        let dropped = drop_pages(&HostFile::Shard(shard), file, Some(map), at)?;
        self.bytes += dropped;
        Ok(())
    }

    /// Release every whole page of the r8 sidecar the pair reads, as
    /// [`PageDrop::release`] does a shard's. Refused by name: a pair that
    /// reads none, and a resident sidecar, whose bytes are an anonymous copy,
    /// which `MADV_DONTNEED` would zero.
    pub fn release_sidecar(&mut self) -> Result<(), PlacementError> {
        let side = self.src.sidecar().ok_or(R8Error::PairReadsNone {
            what: "a sidecar page release",
        })?;
        let map = side.file_pages().ok_or_else(|| R8Error::ResidentCopy {
            path: side.path().to_path_buf(),
        })?;
        let name = HostFile::Sidecar(side.path().to_path_buf());
        let file = File::open(side.path())
            .map_err(|e| PlacementError::Host(format!("open {name} to release pages: {e}")))?;
        self.bytes += drop_pages(&name, &file, Some(map), 0..map.len())?;
        Ok(())
    }

    /// Bytes released so far, whole pages.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn file(&mut self, shard: usize) -> Result<&File, PlacementError> {
        let path =
            self.src.split().shard_path(shard).ok_or_else(|| {
                PlacementError::Host(format!("shard {shard} is not in the split"))
            })?;
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

/// A file's own read-only `MAP_SHARED` mapping, file offset `o` at index `o`:
/// the one kind of mapping [`drop_pages`] drops page-table entries of, where
/// the next touch re-faults the same file bytes. Made by
/// [`FileMapping::shard`] (a split's shard) and `Sidecar::file_pages` (a
/// sidecar opened as its file's mapping); an anonymous copy never becomes
/// one, since `MADV_DONTNEED` would zero it under a live borrow.
#[derive(Clone, Copy)]
pub(crate) struct FileMapping<'a>(&'a [u8]);

impl<'a> FileMapping<'a> {
    /// # Safety
    ///
    /// `map` is the whole of a file's own read-only `MAP_SHARED` mapping,
    /// file offset `o` at index `o`, and no private or anonymous page lies
    /// in it.
    pub(crate) unsafe fn new(map: &'a [u8]) -> FileMapping<'a> {
        FileMapping(map)
    }

    /// Shard `s` of `split`'s mapping; a shard that is no file's mapping (a
    /// header alone, `Split::header_only`) is refused by name.
    pub(crate) fn shard(split: &'a Split, s: usize) -> Result<FileMapping<'a>, PlacementError> {
        let g = shard_of(split, s)?;
        if !g.file_backed() {
            return Err(PlacementError::Host(format!(
                "shard {s} is a header alone, no file's mapping: it holds no page of the file"
            )));
        }
        // SAFETY: a file-backed shard's mapping is the file's own read-only
        // mapping (`Gguf::file_backed`, `Split::open`'s `Gguf::open`).
        Ok(unsafe { FileMapping::new(g.mapping()) })
    }

    /// Its length in bytes.
    pub(crate) fn len(self) -> u64 {
        self.0.len() as u64
    }
}

/// Drop the whole pages inside bytes `at` (file offsets) of `name`'s file
/// `file`: from this process's page tables first when `map` is given, then
/// from the page cache, which skips a page still mapped; the bytes dropped.
/// `map` is the file's own mapping ([`FileMapping`]); `None` is a file this
/// process maps no page of there (a resident sidecar), and only the page
/// cache's copy goes. The one owner of page release: [`PageDrop`] and
/// `r8file::verify` call it, and [`release_file_pages`] for a file outside a
/// split.
pub(crate) fn drop_pages(
    name: &dyn fmt::Display,
    file: &File,
    map: Option<FileMapping<'_>>,
    at: Range<u64>,
) -> Result<u64, PlacementError> {
    let page = page_bytes()?;
    let first = at.start.div_ceil(page) * page;
    let end = at.end / page * page;
    if end <= first {
        return Ok(0);
    }
    if let Some(FileMapping(map)) = map {
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
        // mapping (`FileMapping`'s one invariant). MADV_DONTNEED on it drops
        // this process's page-table entries only; the next touch re-faults
        // the same file bytes, so no borrow of the mapping ever sees other
        // contents.
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
    }
    let off = |v: u64| {
        libc::off_t::try_from(v)
            .map_err(|_| PlacementError::Host(format!("{name} offset {v} passes off_t")))
    };
    drop_cached_pages(file.as_raw_fd(), off(first)?, off(end - first)?).map_err(|e| {
        PlacementError::Host(format!(
            "posix_fadvise(DONTNEED) of {name} bytes {first}..{end}: {e}"
        ))
    })?;
    Ok(end - first)
}

/// [`PageDrop::release`] for a file outside a split (a checkpoint folder's
/// shard): the whole pages inside bytes `at` of the file at `path`, whose
/// whole mapping is `map`, dropped from this process's page tables and then
/// from the page cache; the bytes dropped. For bytes a load has copied that
/// no later reader takes from the file.
///
/// # Safety
///
/// `map` is the whole of the file's own read-only `MAP_SHARED` mapping, file
/// offset `o` at index `o` ([`FileMapping::new`]'s contract).
pub unsafe fn release_file_pages(
    path: &std::path::Path,
    map: &[u8],
    at: Range<u64>,
) -> Result<u64, PlacementError> {
    let file = File::open(path)
        .map_err(|e| PlacementError::Host(format!("open {}: {e}", path.display())))?;
    // SAFETY: the caller's contract is `FileMapping::new`'s.
    let map = unsafe { FileMapping::new(map) };
    drop_pages(&path.display(), &file, Some(map), at)
}

/// The line a page drop prints once when the platform has no
/// `posix_fadvise(DONTNEED)`: the page cache keeps the pages the call would
/// have dropped.
const NO_FADVISE: &str = "host pages: this platform has no posix_fadvise(DONTNEED), so the page \
                          cache keeps the pages";

/// Whether the line has been printed already.
static SAID_FADVISE: AtomicBool = AtomicBool::new(false);

/// Drop the clean pages of `fd` over `at .. at + len` from the page cache,
/// where the platform has such a call. A platform without one keeps its
/// pages: the refusal is the platform saying so, the line says it once, and
/// the drop goes on. Any other error is the caller's to refuse.
pub(crate) fn drop_cached_pages(
    fd: std::os::fd::RawFd,
    at: libc::off_t,
    len: libc::off_t,
) -> Result<(), os::OsError> {
    match os::fadvise_dontneed(fd, at, len) {
        Ok(()) => Ok(()),
        Err(os::OsError::Unsupported {
            call: "posix_fadvise",
        }) => {
            if !SAID_FADVISE.swap(true, Ordering::Relaxed) {
                eprintln!("{NO_FADVISE}");
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// The line a populate prints once when the kernel has no
/// `MADV_POPULATE_READ`.
#[cfg(target_os = "linux")]
pub(crate) const NO_POPULATE_READ: &str = "host populate: madvise(MADV_POPULATE_READ) is not in this kernel \
                                (EINVAL; it came in Linux 5.14), so the load reads one byte a page \
                                instead";

/// The line a populate prints once when the platform has no
/// `MADV_POPULATE_READ`.
#[cfg(target_os = "macos")]
pub(crate) const NO_POPULATE_READ: &str = "host populate: this platform has no madvise(MADV_POPULATE_READ), \
                                so the load reads one byte a page instead";

/// `madvise(MADV_POPULATE_READ)` over `span`, a live sub-slice of a set's
/// mappings: a shard's read-only file mapping, or the sidecar's, which is its
/// file's mapping or a resident anonymous copy.
pub(crate) fn advise_populate(span: &[u8]) -> Result<(), os::OsError> {
    // SAFETY: `span` is a live slice (the caller's contract above), so its
    // pages are inside one live mapping.
    unsafe { os::advise_populate(span.as_ptr().cast_mut().cast(), span.len()) }
}

/// Fault the pages of `span` in through `advise` (the madvise call,
/// [`advise_populate`]); a kernel that does not know the advice (EINVAL,
/// before Linux 5.14) or a platform without it ([`os::OsError::Unsupported`])
/// gets the same pages faulted by a read of one byte each `page` bytes
/// instead. Any other refusal is the walk's, by name. The error stays typed
/// through this seam: the fallback reads the variant and the errno.
pub(crate) fn populate_span(
    span: &[u8],
    page: u64,
    advise: impl Fn(&[u8]) -> Result<(), os::OsError>,
) -> Result<Populated, String> {
    // SAFETY: `span` is a live slice (the caller's contract above), so its
    // pages are inside one live mapping.
    unsafe {
        os::populate_span(span.as_ptr().cast_mut().cast(), span.len(), page, || {
            advise(span)
        })
    }
    .map_err(|e| format!("madvise(MADV_POPULATE_READ): {e}"))
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

/// Run `op` over every page run of `set` in `src`'s mappings — the split's
/// and the sidecar's — one thread per chunk ([`chunks`]), a chunk's runs in
/// order. `op` returns a count each chunk sums (the resident pages of a residency
/// count; 0 for a populate or a lock). Returns the spans `op` succeeded on
/// and the walk, or the first refusal in chunk order.
fn walk(
    src: R8Source<'_>,
    set: &HostSet,
    op: impl Fn(&[u8]) -> Result<u64, String> + Sync,
) -> (Vec<(usize, usize)>, Result<Walked, PlacementError>) {
    let work = match chunks(src, set) {
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
/// run's bytes. A panic in `op` is caught here and is the chunk's refusal,
/// naming the file and the panic's message, with the spans `op` succeeded on
/// before it still handed back, so that a lock's drop unlocks them.
fn walk_chunk(
    c: &Chunk<'_>,
    page: u64,
    op: &(impl Fn(&[u8]) -> Result<u64, String> + Sync),
) -> ChunkResult {
    let mut done = Vec::with_capacity(c.runs.len());
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        walk_runs(c, page, op, &mut done)
    }));
    let result = walked.unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a payload that is not text".to_string());
        Err(PlacementError::Host(format!(
            "{}: a walk chunk panicked after {} spans: {why}",
            c.file,
            done.len()
        )))
    });
    (done, result)
}

/// [`walk_chunk`]'s loop: each span `op` succeeded on goes to `done` at once.
fn walk_runs(
    c: &Chunk<'_>,
    page: u64,
    op: &(impl Fn(&[u8]) -> Result<u64, String> + Sync),
    done: &mut Vec<(usize, usize)>,
) -> Result<(ChunkWalk, u64), PlacementError> {
    let start = Instant::now();
    let (mut bytes, mut count) = (0, 0);
    for r in &c.runs {
        let span = run_span(c.g, &c.file, r, page)?;
        match op(span) {
            Ok(n) => count += n,
            Err(e) => {
                let at = r.start * page;
                return Err(PlacementError::Host(format!(
                    "{} bytes {at}..{}: {e}",
                    c.file,
                    at + span.len() as u64
                )));
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
    Ok((record, count))
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
    extra: &[(usize, ExpertList)],
    page: u64,
) -> Result<(Vec<Runs>, Runs), PlacementError> {
    let mut pages: Vec<Runs> = vec![Vec::new(); split.shard_count()];
    let mut side: Runs = Vec::new();
    let named = |tensor: usize, what: &str| {
        plan.model
            .tensors
            .get(tensor)
            .ok_or_else(|| PlacementError::Host(format!("{what} {tensor} names no tensor")))
    };
    let mut pool: Vec<(&ModelTensor, Segment)> = Vec::with_capacity(extra.len());
    for (tensor, list) in extra {
        let t = named(*tensor, "an extra host run's tensor")?;
        pool.push((
            t,
            Segment {
                device: Device::Host,
                format: Format::HostFile,
                experts: Some(list.clone()),
                resident_bytes: 0,
            },
        ));
    }
    let mut segments: Vec<(&ModelTensor, &Segment)> = Vec::new();
    for row in &plan.rows {
        let t = named(row.tensor, "plan row")?;
        if !keep(t) {
            continue;
        }
        segments.extend(
            row.segments
                .iter()
                .filter(|seg| seg.device == Device::Host && seg.format == Format::HostFile)
                .map(|seg| (t, seg)),
        );
    }
    segments.extend(pool.iter().map(|(t, seg)| (*t, seg)));
    for &(t, seg) in &segments {
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
        let runs = runs_of(split, t)?;
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
            if let Some(runs) = &runs {
                // The stack's experts lie where the folder stores them: each
                // expert of the span is its own run, not a share of one.
                if held.is_some() {
                    return Err(PlacementError::tensor(
                        t,
                        "is a located stack, which the r8 sidecar cannot hold",
                    ));
                }
                let (_, per) = super::per_expert(t, plan.model.experts_of(t))?;
                let runs = located_pages(runs, per, &span.bytes, page)
                    .map_err(|detail| PlacementError::tensor(t, detail))?;
                pages[s].extend(runs);
                continue;
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
    // `span` starts on a page boundary inside one live mapping (a set run
    // starts at a page multiple of the mapping's page-aligned start), and
    // `vec` has one byte for each of its pages.
    // SAFETY: the caller's contract above.
    unsafe { os::mincore(span.as_ptr().cast_mut().cast(), span.len(), &mut vec) }
        .map_err(io::Error::from)?;
    Ok((vec.iter().filter(|&&b| b & 1 != 0).count() as u64, n))
}

#[cfg(test)]
mod tests {
    use super::{HostSet, Populated, cut, located_pages, populate_span, walk};
    use crate::ops::{RowLayout, StackRuns};
    use crate::placement::PlacementError;
    use crate::r8file::{self, HostR8, R8Error, R8Source, Sidecar};
    use std::fs::File;
    use std::io::BufWriter;
    use std::ops::Range;
    use std::sync::Arc;

    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{Split, Weights};

    /// A kernel without `MADV_POPULATE_READ` (EINVAL) or a platform without
    /// it (`Unsupported`) gets the span's pages read one byte each instead; an
    /// advice the kernel takes reads nothing more, and any other refusal is
    /// named. The madvise call is the seam: the error comes from the test, not
    /// from the running kernel.
    #[test]
    fn a_kernel_without_populate_read_reads_a_byte_a_page() {
        let page = 4096u64;
        let span: Vec<u8> = (0..3 * 4096 + 7).map(|i| (i % 251) as u8).collect();
        let einval = |_: &[u8]| {
            Err(os::OsError::Io(std::io::Error::from_raw_os_error(
                libc::EINVAL,
            )))
        };
        assert_eq!(populate_span(&span, page, einval), Ok(Populated::Touched));
        let absent = |_: &[u8]| {
            Err(os::OsError::Unsupported {
                call: "madvise(MADV_POPULATE_READ)",
            })
        };
        assert_eq!(populate_span(&span, page, absent), Ok(Populated::Touched));
        assert_eq!(
            populate_span(&span, page, |_: &[u8]| Ok(())),
            Ok(Populated::Advised)
        );
        let enomem = |_: &[u8]| {
            Err(os::OsError::Io(std::io::Error::from_raw_os_error(
                libc::ENOMEM,
            )))
        };
        let e = populate_span(&span, page, enomem).unwrap_err();
        assert!(e.starts_with("madvise(MADV_POPULATE_READ): "), "{e}");
        assert_eq!(
            populate_span(&[], page, einval),
            Ok(Populated::Touched),
            "an empty span reads nothing"
        );
    }

    /// A host set built beside one open of a sidecar walks only beside a pair
    /// that reads that open: a pair that reads none, or another open of the
    /// same file checked against the same split, is refused by name
    /// (`R8Error::OtherSidecar`) before a page is walked. One Q3_K stack of
    /// two experts of 8 rows of 256, its sidecar converted beside it.
    /// A known run is held and never walked: `holds` answers for a range
    /// whose pages lie in the known runs, across the page a walked run
    /// shares with them too, while the set's pages and runs, which every
    /// populate and lock walks, are the walked ones alone.
    #[test]
    fn a_known_run_is_held_and_not_walked() {
        let page = super::page_bytes().unwrap();
        let set = HostSet {
            page,
            shards: vec![(0, std::iter::once(0..2).collect())],
            side: None,
            held: Some((vec![std::iter::once(0..5).collect()], Vec::new())),
        };
        let shard = super::HostFile::Shard(0);
        assert!(set.holds(&shard, &(page..2 * page)));
        assert!(set.holds(&shard, &(page..4 * page)));
        assert!(set.holds(&shard, &(3 * page..5 * page)));
        assert!(!set.holds(&shard, &(4 * page..6 * page)));
        assert!(!set.holds(&super::HostFile::Shard(1), &(0..page)));
        assert_eq!(set.pages(), 2);
        assert_eq!(set.runs().len(), 1);
        assert_eq!(set.runs()[0].1.len(), 1);
        assert_eq!(set.runs()[0].1.first(), Some(&(0..2)));
    }

    #[test]
    fn a_set_walks_only_beside_its_own_sidecar() {
        let page = super::page_bytes().unwrap();
        let dir = std::env::temp_dir().join(format!("host-lock-pair-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        let nbytes = 2 * 8 * 110;
        let decl = TensorDecl {
            name: "s".to_string(),
            dims: vec![256, 8, 2],
            type_id: 11,
            nbytes,
        };
        let layout = Layout::new(&[], vec![decl]).unwrap();
        let mut w = Writer::new(BufWriter::new(File::create(&path).unwrap()), layout).unwrap();
        let bytes: Vec<u8> = (0..nbytes).map(|i| (i * 7 % 251) as u8).collect();
        w.tensor("s", &bytes).unwrap();
        w.finish().unwrap();
        let split = Split::open(&path).unwrap();
        let side_path = dir.join("m-r8.gguf");
        r8file::convert(&split, &["s".to_string()], &side_path, &mut |_| {}).unwrap();
        let open = || {
            let lazy = Weights::Mapped { populate: false };
            HostR8::On(Arc::new(Sidecar::open(&side_path, &split, lazy).unwrap()))
        };
        let (held, other) = (open(), open());
        let side = held.sidecar().unwrap();
        let set = HostSet {
            page,
            shards: Vec::new(),
            side: Some((Arc::clone(side), std::iter::once(0..1).collect())),
            held: None,
        };
        for (what, src) in [
            ("a pair that reads none", R8Source::rows(&split)),
            (
                "a pair that reads another open",
                R8Source::of(&split, &other).unwrap(),
            ),
        ] {
            match set.resident(src) {
                Err(PlacementError::R8(e @ R8Error::OtherSidecar { .. })) => {
                    println!("walk beside {what}: {e}");
                }
                other => panic!("walk beside {what} must be refused by name, got {other:?}"),
            }
        }
        let files = set.resident(R8Source::of(&split, &held).unwrap()).unwrap();
        assert_eq!(files.len(), 1, "the set's one file");
        assert_eq!(files[0].pages, 1, "the set's one page");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A walk whose `op` panics in a chunk hands back the span that chunk
    /// took before the panic — so a lock's drop unlocks it — and refuses by
    /// name, saying the chunk panicked. One file, 32 single-page runs two
    /// pages apart: 16 chunks of two runs; `op` panics on chunk 0's second.
    #[test]
    fn a_panicking_walk_chunk_hands_back_its_spans() {
        let page = super::page_bytes().unwrap();
        let dir = std::env::temp_dir().join(format!("host-lock-panic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.gguf");
        let n = usize::try_from(66 * page / 4).unwrap();
        let decl = TensorDecl {
            name: "w".to_string(),
            dims: vec![n as u64],
            type_id: 0,
            nbytes: 4 * n as u64,
        };
        let layout = Layout::new(&[], vec![decl]).unwrap();
        let mut w = Writer::new(BufWriter::new(File::create(&path).unwrap()), layout).unwrap();
        w.tensor("w", &vec![0u8; 4 * n]).unwrap();
        w.finish().unwrap();
        let split = Split::open(&path).unwrap();
        let runs: Vec<Range<u64>> = (0..32).map(|i| 2 * i..2 * i + 1).collect();
        let set = HostSet {
            page,
            shards: vec![(0, runs)],
            side: None,
            held: None,
        };
        let map = split.shard(0).unwrap().mapping().as_ptr() as usize;
        let page_len = usize::try_from(page).unwrap();
        let second = map + 2 * page_len;
        let (spans, walked) = walk(R8Source::rows(&split), &set, |span| {
            assert!(
                span.as_ptr() as usize != second,
                "the op refuses page 2 by panicking"
            );
            Ok(0)
        });
        let e = walked.map(|_| ()).unwrap_err().to_string();
        assert!(e.contains("panicked"), "the refusal names the panic: {e}");
        assert!(
            spans.contains(&(map, page_len)),
            "the panicking chunk's first span comes back for the unlock: {spans:?}"
        );
        assert_eq!(
            spans.len(),
            31,
            "every span but the panicked one comes back"
        );
        std::fs::remove_dir_all(&dir).unwrap();
        println!("walk: a chunk's panic is its named refusal, its taken span handed back ({e})");
    }

    /// A header alone (`Split::header_only`) is no file's mapping: a page release over it is
    /// refused by name before any advice, as its bytes are a heap copy `MADV_DONTNEED` would zero.
    #[test]
    fn a_header_alone_releases_no_page() {
        let decl = TensorDecl {
            name: "w".to_string(),
            dims: vec![4],
            type_id: 0,
            nbytes: 16,
        };
        let layout = Layout::new(&[], vec![decl]).unwrap();
        let split = Split::header_only(std::env::temp_dir().join("header-alone"), &layout).unwrap();
        let e = super::PageDrop::new(&split)
            .release(0, 0..16)
            .unwrap_err()
            .to_string();
        assert!(e.contains("a header alone"), "{e}");
    }

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

    /// A located stack's host segment is a range of pages for each run of
    /// each of its experts — a planes expert's codes then its scales —
    /// wherever the folder stores them: experts 1 and 2 of a stack whose
    /// experts lie out of order lock exactly their own runs' pages, not the
    /// pages an equal-stride stack's experts 1 and 2 would, and an expert's
    /// scales are pages of their own where the folder keeps them a region
    /// away. A span that cuts an expert, or names one past the runs, is
    /// refused by name.
    #[test]
    fn a_located_stacks_host_span_is_the_pages_of_its_experts_runs() {
        let page = 4096u64;
        let (cp, sp) = (2 * page + 512, page / 2 + 64);
        let per = cp + sp;
        let codes = vec![40, 10 * page + 24, 5 * page + 8, 20 * page];
        let region = 60 * page;
        let separated = vec![
            region + 3,
            region + 9 * page,
            region + 5 * page + 100,
            region + 7,
        ];
        let adjacent: Vec<u64> = codes.iter().map(|&c| c + cp).collect();
        for (what, scales) in [("separated", separated), ("adjacent", adjacent)] {
            let runs =
                StackRuns::new(RowLayout::Planes, codes.clone(), scales.clone(), cp, sp).unwrap();
            let pages = |a: u64, len: u64| a / page..(a + len).div_ceil(page);
            let expert = |e: usize| [pages(codes[e], cp), pages(scales[e], sp)];
            assert_eq!(
                located_pages(&runs, per, &(per..3 * per), page).unwrap(),
                [expert(1), expert(2)].concat(),
                "{what}: experts 1 and 2: the pages of the runs the offsets name, codes then scales"
            );
            let equal_stride = (codes[0] + per) / page..(codes[0] + 3 * per).div_ceil(page);
            assert!(
                !located_pages(&runs, per, &(per..3 * per), page)
                    .unwrap()
                    .contains(&equal_stride),
                "{what}: not the pages of one run from the first expert's offset"
            );
            assert_eq!(
                located_pages(&runs, per, &(0..4 * per), page).unwrap(),
                (0..4).flat_map(expert).collect::<Vec<_>>(),
                "{what}: the whole stack: every expert's runs"
            );
            let cut = located_pages(&runs, per, &(per + 1..2 * per), page).unwrap_err();
            assert!(
                cut.contains("not whole experts of the 4 located runs"),
                "{what}: {cut}"
            );
            assert!(located_pages(&runs, per, &(0..5 * per), page).is_err());
        }
        let rows = StackRuns::new(RowLayout::Rows, codes.clone(), Vec::new(), per, 0).unwrap();
        assert_eq!(
            located_pages(&rows, per, &(per..2 * per), page).unwrap(),
            vec![codes[1] / page..(codes[1] + per).div_ceil(page)],
            "a rows stack is one run an expert"
        );
    }
}
