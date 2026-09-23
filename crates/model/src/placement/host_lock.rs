//! The host tier held in RAM: the file bytes of the plan's host segments,
//! read and locked where they lie — in the shards' own mappings, nothing
//! copied. One derivation, [`HostSet`], names those pages; a load populates
//! them ([`HostSet::populate`]) so that no decode step takes the first-touch
//! read, and a serving process may also lock them ([`HostLock`]) so that no
//! other reader of the page cache can evict an expert page from under a step.
//! Both walk the same set, so the populated bytes and the locked bytes are
//! the same bytes by construction.
//!
//! The complement lives here too: [`PageDrop`] releases file bytes the load
//! has put on a card and no later reader needs, so that the upload does not
//! leave them in the page cache in place of the host tier's.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::thread;
use std::time::{Duration, Instant};

use gguf::{Gguf, Split};

use super::{Device, Format, ModelTensor, PlacementError, Plan};

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

/// The pages of every host segment a plan keeps, per shard: page ranges
/// `[first, end)` of the shard's mapping, sorted and merged where they touch.
#[derive(Clone, Debug)]
pub struct HostSet {
    page: u64,
    /// Only shards with at least one range, in shard order.
    shards: Vec<(usize, Vec<Range<u64>>)>,
}

/// One shard's side of a walk over a [`HostSet`] — a populate or a lock.
#[derive(Clone, Debug)]
pub struct ShardWalk {
    pub shard: usize,
    /// Page spans walked: the host segments' ranges, merged where they touch.
    pub spans: usize,
    /// Bytes walked, whole pages.
    pub bytes: u64,
    /// This shard's calls end to end — the reads of the pages that were not
    /// in the page cache.
    pub wall: Duration,
}

/// A walk over every shard of a [`HostSet`], one thread per shard.
#[derive(Clone, Debug)]
pub struct Walk {
    shards: Vec<ShardWalk>,
    wall: Duration,
}

impl Walk {
    /// Bytes walked, whole pages, over every shard.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes).sum()
    }

    /// Per shard, in shard order.
    #[must_use]
    pub fn shards(&self) -> &[ShardWalk] {
        &self.shards
    }

    /// The whole walk end to end, all shards' threads.
    #[must_use]
    pub fn wall(&self) -> Duration {
        self.wall
    }
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
        let page = page_bytes();
        let shards = host_pages(split, plan, keep, page)?
            .into_iter()
            .enumerate()
            .filter(|(_, runs)| !runs.is_empty())
            .collect();
        Ok(HostSet { page, shards })
    }

    /// Bytes of the set, whole pages.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pages() * self.page
    }

    /// Pages of the set.
    #[must_use]
    pub fn pages(&self) -> u64 {
        self.shards
            .iter()
            .flat_map(|(_, runs)| runs)
            .map(|r| r.end - r.start)
            .sum()
    }

    /// Read every page of the set into the page cache and map it in `split`'s
    /// mappings (`MADV_POPULATE_READ`), one thread per shard. A page already
    /// cached costs its page-table entry only.
    pub fn populate(&self, split: &Split) -> Result<Walk, PlacementError> {
        walk_shards(split, self, |span| {
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
                Ok(())
            } else {
                Err(format!(
                    "madvise(MADV_POPULATE_READ): {}",
                    io::Error::last_os_error()
                ))
            }
        })
        .1
    }

    /// Pages of the set `mincore` reports resident in the page cache, and
    /// all of them. `split` must be the split the set was made from.
    pub fn resident(&self, split: &Split) -> Result<(u64, u64), PlacementError> {
        let (mut resident, mut total) = (0, 0);
        for (s, runs) in &self.shards {
            let g = shard_of(split, *s)?;
            for r in runs {
                let span = run_span(g, *s, r, self.page)?;
                let (n_in, n) = resident_pages(span, self.page).map_err(|e| {
                    PlacementError::Host(format!("mincore over shard {s} pages {r:?}: {e}"))
                })?;
                resident += n_in;
                total += n;
            }
        }
        Ok((resident, total))
    }
}

/// The pages of a [`HostSet`], locked in the shards' mappings until this
/// drops. Owned: it holds the spans' addresses, not a borrow of the split,
/// so the owner of the split can hold it too.
///
/// The split's mappings must outlive the lock (drop the lock first). If they
/// go first, the unmapping has already released the pages and the drop's
/// `munlock` over the old addresses changes only residency — never memory
/// contents.
pub struct HostLock {
    /// `(address, length)` of every span locked.
    spans: Vec<(usize, usize)>,
    walk: Walk,
}

impl HostLock {
    /// Lock every page of `set` in `split`'s mappings, one thread per shard.
    /// A page not yet in the page cache is read first, so a set populated
    /// just before locks at page-table speed. A refusal — `RLIMIT_MEMLOCK`,
    /// most often — unlocks what was taken and names the limit.
    pub fn lock(split: &Split, set: &HostSet) -> Result<HostLock, PlacementError> {
        let (spans, walk) = walk_shards(split, set, |span| {
            // SAFETY: `span` is a live sub-slice of a read-only file mapping;
            // mlock faults its pages in and pins them, and never writes them.
            if unsafe { libc::mlock(span.as_ptr().cast(), span.len()) } == 0 {
                Ok(())
            } else {
                let e = io::Error::last_os_error();
                Err(format!("mlock: {e} ({})", memlock_limit()))
            }
        });
        let mut lock = HostLock {
            spans,
            walk: Walk {
                shards: Vec::new(),
                wall: Duration::ZERO,
            },
        };
        // An error drops the partial lock, which unlocks what was taken.
        lock.walk = walk?;
        Ok(lock)
    }

    /// Bytes locked, whole pages, over every shard.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.walk.bytes()
    }

    /// Per shard, in shard order.
    #[must_use]
    pub fn shards(&self) -> &[ShardWalk] {
        self.walk.shards()
    }

    /// The whole lock end to end, all shards' threads.
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
        let first = at.start.div_ceil(self.page) * self.page;
        let end = at.end / self.page * self.page;
        if end <= first {
            return Ok(());
        }
        let g = shard_of(self.split, shard)?;
        let map = g.mapping();
        let span = usize::try_from(first)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(a, b)| map.get(a..b))
            .ok_or_else(|| {
                PlacementError::Host(format!(
                    "shard {shard} bytes {at:?} run past its {} bytes",
                    map.len()
                ))
            })?;
        // SAFETY: `span` is a live sub-slice of a read-only MAP_SHARED file
        // mapping. MADV_DONTNEED on it drops this process's page-table
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
                "madvise(MADV_DONTNEED) of shard {shard} bytes {first}..{end}: {}",
                io::Error::last_os_error()
            )));
        }
        let file = self.file(shard)?;
        let off = |v: u64| {
            libc::off_t::try_from(v)
                .map_err(|_| PlacementError::Host(format!("shard {shard} offset {v} passes off_t")))
        };
        // SAFETY: the descriptor is this shard file's own and lives in `self`
        // across the call; fadvise reads no memory of ours.
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
                "posix_fadvise(DONTNEED) of shard {shard} bytes {first}..{end}: {}",
                io::Error::from_raw_os_error(rc)
            )));
        }
        self.bytes += end - first;
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

/// What one shard's thread hands back: every span its call succeeded on —
/// also when a later one failed, so that a lock's drop unlocks them — and
/// its record.
type ShardResult = (Vec<(usize, usize)>, Result<ShardWalk, PlacementError>);

/// Run `op` over every page run of `set` in `split`'s mappings, one thread
/// per shard, runs in order within a shard. Returns the spans `op` succeeded
/// on and the walk, or the first shard's refusal.
fn walk_shards(
    split: &Split,
    set: &HostSet,
    op: impl Fn(&[u8]) -> Result<(), String> + Sync,
) -> (Vec<(usize, usize)>, Result<Walk, PlacementError>) {
    let mut work = Vec::with_capacity(set.shards.len());
    for (s, runs) in &set.shards {
        match shard_of(split, *s) {
            Ok(g) => work.push((*s, g, runs)),
            Err(e) => return (Vec::new(), Err(e)),
        }
    }
    let start = Instant::now();
    let op = &op;
    let results: Vec<ShardResult> = thread::scope(|scope| {
        let handles: Vec<_> = work
            .iter()
            .map(|&(s, g, runs)| scope.spawn(move || walk_shard(g, s, runs, set.page, op)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    let e = PlacementError::Host("a shard's walk thread panicked".into());
                    (Vec::new(), Err(e))
                })
            })
            .collect()
    });
    let wall = start.elapsed();
    let mut done = Vec::new();
    let mut shards = Vec::with_capacity(results.len());
    let mut refused = None;
    for (spans, shard) in results {
        done.extend(spans);
        match shard {
            Ok(s) => shards.push(s),
            Err(e) => refused = refused.or(Some(e)),
        }
    }
    match refused {
        Some(e) => (done, Err(e)),
        None => (done, Ok(Walk { shards, wall })),
    }
}

/// `op` over `runs` of shard `shard`'s mapping, in order.
fn walk_shard(
    g: &Gguf,
    shard: usize,
    runs: &[Range<u64>],
    page: u64,
    op: &(impl Fn(&[u8]) -> Result<(), String> + Sync),
) -> ShardResult {
    let start = Instant::now();
    let mut done = Vec::with_capacity(runs.len());
    let mut bytes = 0;
    for r in runs {
        let span = match run_span(g, shard, r, page) {
            Ok(span) => span,
            Err(e) => return (done, Err(e)),
        };
        if let Err(e) = op(span) {
            let e = PlacementError::Host(format!("shard {shard} pages {r:?}: {e}"));
            return (done, Err(e));
        }
        done.push((span.as_ptr() as usize, span.len()));
        bytes += (r.end - r.start) * page;
    }
    let record = ShardWalk {
        shard,
        spans: runs.len(),
        bytes,
        wall: start.elapsed(),
    };
    (done, Ok(record))
}

fn shard_of(split: &Split, s: usize) -> Result<&Gguf, PlacementError> {
    split
        .shard(s)
        .ok_or_else(|| PlacementError::Host(format!("shard {s} is not in the split")))
}

/// Pages `r` of `g`'s mapping as a slice, the last one clamped to the
/// mapping's end.
fn run_span<'g>(
    g: &'g Gguf,
    shard: usize,
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
                "shard {shard} pages {r:?} run past its {} bytes",
                map.len()
            ))
        })
}

/// Per shard, the page ranges `[first, end)` of the host segments `keep`
/// selects, sorted and merged where they overlap or touch.
fn host_pages(
    split: &Split,
    plan: &Plan<'_>,
    keep: impl Fn(&ModelTensor) -> bool,
    page: u64,
) -> Result<Vec<Vec<Range<u64>>>, PlacementError> {
    let mut pages: Vec<Vec<Range<u64>>> = vec![Vec::new(); split.shard_count()];
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
                let base = g.data_base() + info.offset;
                let (a, b) = (base + span.bytes.start, base + span.bytes.end);
                pages[s].push(a / page..b.div_ceil(page));
            }
        }
    }
    for spans in &mut pages {
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
    Ok(pages)
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
