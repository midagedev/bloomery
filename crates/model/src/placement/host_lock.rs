//! The host tier held in RAM: the file bytes of the plan's host segments,
//! locked where they lie — in the shards' own mappings, nothing copied — so
//! that no other reader of the page cache can evict an expert page from under
//! a decode step. Off unless asked: a serving process takes a [`HostLock`];
//! gates and measurements do not.

use std::io;
use std::ops::Range;
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

/// One shard's side of a [`HostLock`].
#[derive(Clone, Debug)]
pub struct ShardLock {
    pub shard: usize,
    /// Page spans locked: the host segments' ranges, merged where they touch.
    pub spans: usize,
    /// Bytes locked, whole pages.
    pub bytes: u64,
    /// This shard's `mlock` calls end to end — the reads of the pages that
    /// were not in the page cache.
    pub wall: Duration,
}

/// The pages of every chosen host segment, locked in the shards' mappings
/// until this drops.
pub struct HostLock<'a> {
    spans: Vec<&'a [u8]>,
    shards: Vec<ShardLock>,
    wall: Duration,
}

/// What one shard's thread hands back: every span it locked — also when a
/// later one failed, so that dropping the lock unlocks them — and its record.
type ShardResult<'a> = (Vec<&'a [u8]>, Result<ShardLock, PlacementError>);

impl<'a> HostLock<'a> {
    /// Lock the file bytes of every segment `plan` puts on the host in the
    /// file's format whose tensor `keep` selects: each run of consecutive
    /// experts of an expert stack's list, or the whole tensor. Each run grows
    /// outward to whole pages, the pages merge per shard, and one thread per
    /// shard locks them. `plan` must be a plan of `split`'s tensors.
    pub fn lock(
        split: &'a Split,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
    ) -> Result<HostLock<'a>, PlacementError> {
        let page = page_bytes();
        let mut work = Vec::new();
        for (s, pages) in host_pages(split, plan, keep, page)?.into_iter().enumerate() {
            if pages.is_empty() {
                continue;
            }
            let g = split
                .shard(s)
                .ok_or_else(|| PlacementError::Host(format!("shard {s} is not in the split")))?;
            work.push((s, g, pages));
        }
        let start = Instant::now();
        let results: Vec<ShardResult<'a>> = thread::scope(|scope| {
            let handles: Vec<_> = work
                .iter()
                .map(|&(s, g, ref pages)| scope.spawn(move || lock_shard(g, s, pages, page)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        let e = PlacementError::Host("a shard's lock thread panicked".into());
                        (Vec::new(), Err(e))
                    })
                })
                .collect()
        });
        let mut lock = HostLock {
            spans: Vec::new(),
            shards: Vec::new(),
            wall: start.elapsed(),
        };
        let mut refused = None;
        for (spans, shard) in results {
            lock.spans.extend(spans);
            match shard {
                Ok(s) => lock.shards.push(s),
                Err(e) => refused = refused.or(Some(e)),
            }
        }
        // A refusal drops the partial lock, which unlocks what was taken.
        match refused {
            Some(e) => Err(e),
            None => Ok(lock),
        }
    }

    /// Bytes locked, whole pages, over every shard.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes).sum()
    }

    /// Per shard, in shard order.
    #[must_use]
    pub fn shards(&self) -> &[ShardLock] {
        &self.shards
    }

    /// The whole lock end to end, all shards' threads.
    #[must_use]
    pub fn wall(&self) -> Duration {
        self.wall
    }

    /// Pages of the locked spans `mincore` reports resident, and all of them.
    pub fn resident(&self) -> Result<(u64, u64), PlacementError> {
        let page = page_bytes();
        let (mut resident, mut total) = (0, 0);
        for span in &self.spans {
            let (r, n) = resident_pages(span, page)
                .map_err(|e| PlacementError::Host(format!("mincore over a locked span: {e}")))?;
            resident += r;
            total += n;
        }
        Ok((resident, total))
    }
}

impl Drop for HostLock<'_> {
    fn drop(&mut self) {
        for span in &self.spans {
            // SAFETY: `span` is a live sub-slice of a shard's mapping, borrowed
            // for this lock's lifetime; munlock changes only its pages'
            // residency. A failure leaves them locked until the mapping goes,
            // and a drop has no one to report it to.
            unsafe {
                libc::munlock(span.as_ptr().cast(), span.len());
            }
        }
    }
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
                    "is not in the split the lock maps",
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

/// Lock `pages` of shard `shard`'s mapping, in order.
fn lock_shard<'a>(g: &'a Gguf, shard: usize, pages: &[Range<u64>], page: u64) -> ShardResult<'a> {
    let map = g.mapping();
    let start = Instant::now();
    let mut locked = Vec::with_capacity(pages.len());
    let mut bytes = 0;
    for r in pages {
        let first = r.start * page;
        let end = (r.end * page).min(map.len() as u64);
        let span = usize::try_from(first)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(a, b)| map.get(a..b));
        let Some(span) = span else {
            let e = PlacementError::Host(format!(
                "shard {shard} pages {r:?} run past its {} bytes",
                map.len()
            ));
            return (locked, Err(e));
        };
        // SAFETY: `span` is a live sub-slice of this shard's mapping, borrowed
        // for `'a`; mlock faults its pages in and pins them, and never writes
        // them.
        if unsafe { libc::mlock(span.as_ptr().cast(), span.len()) } != 0 {
            let e = PlacementError::Host(format!(
                "mlock of shard {shard} pages {r:?}: {}",
                io::Error::last_os_error()
            ));
            return (locked, Err(e));
        }
        locked.push(span);
        bytes += (r.end - r.start) * page;
    }
    let record = ShardLock {
        shard,
        spans: pages.len(),
        bytes,
        wall: start.elapsed(),
    };
    (locked, Ok(record))
}

/// `mincore` over `span`: its resident pages and all of them.
fn resident_pages(span: &[u8], page: u64) -> io::Result<(u64, u64)> {
    let n = (span.len() as u64).div_ceil(page);
    let mut vec = vec![0u8; usize::try_from(n).map_err(io::Error::other)?];
    // SAFETY: `span` starts on a page boundary inside one live mapping (a lock
    // span starts at a page multiple of the mapping's page-aligned start), and
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
