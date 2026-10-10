//! Pinned host memory for sequence states saved off the request path
//! ([`Staging`]): a pool of page-locked slabs, lent to a state as one extent
//! ([`Staged`]) and taken back when the state is dropped.
//!
//! Page-locking costs far more than a copy over the link, so a save never
//! allocates for itself once the pool has grown to the states it holds: a
//! state's extent is cut from a slab the pool already has, and the extent of
//! a state the cache evicted is the next one's. The pool grows a slab at a
//! time, up to its limit (the prompt cache's budget), and a request past the
//! limit is refused by name. What a growth cost is recorded
//! ([`Staged::alloc_ms`], [`StagingStats`]), apart from the copy.
//!
//! An extent is never lent again while a copy writes it: the copy's event is
//! held with the extent ([`Staged::hold`]), and dropping the extent waits for
//! it first.

use std::ops::Range;
use std::slice;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use cuda_core::{CudaContext, CudaEvent, PinnedHostBuffer};
use runtime::savemap::{Extents, Layout};

use crate::GpuError;

/// A slab's size at least: the pool grows by this much, or by the state's
/// own bytes when it needs more.
const SLAB: usize = 64 << 20;

/// An extent starts on a boundary of this many bytes, a huge page's.
const GRAIN: usize = 2 << 20;

const WHAT: &str = "staging";

struct Slab {
    mem: PinnedHostBuffer<u8>,
    extents: Extents,
}

struct Pool {
    slabs: Vec<Slab>,
    /// Pinned bytes the slabs hold, lent or not.
    held: usize,
    /// Bytes of the extents lent now.
    lent: usize,
    /// Slabs allocated, and the ms they took, for the pool's life.
    grown: u64,
    grown_ms: f64,
}

struct Shared {
    ctx: Arc<CudaContext>,
    limit: usize,
    pool: Mutex<Pool>,
}

impl Shared {
    fn pool(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.pool.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What the pool holds and has cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StagingStats {
    /// Pinned bytes in the slabs.
    pub held: u64,
    /// Bytes of the extents lent now.
    pub lent: u64,
    /// The pool's limit.
    pub limit: u64,
    /// Slabs allocated, and the ms their allocation took, in all.
    pub grown: u64,
    pub grown_ms: f64,
}

/// The pool: one per model, cloned handles share it.
#[derive(Clone)]
pub struct Staging {
    shared: Arc<Shared>,
}

impl Staging {
    /// An empty pool of at most `limit` pinned bytes on `ctx`'s device. It
    /// allocates nothing until a state asks.
    #[must_use]
    pub fn new(ctx: &Arc<CudaContext>, limit: u64) -> Staging {
        Staging {
            shared: Arc::new(Shared {
                ctx: Arc::clone(ctx),
                limit: usize::try_from(limit).unwrap_or(usize::MAX),
                pool: Mutex::new(Pool {
                    slabs: Vec::new(),
                    held: 0,
                    lent: 0,
                    grown: 0,
                    grown_ms: 0.0,
                }),
            }),
        }
    }

    /// What the pool holds and has cost.
    #[must_use]
    pub fn stats(&self) -> StagingStats {
        let p = self.shared.pool();
        StagingStats {
            held: p.held as u64,
            lent: p.lent as u64,
            limit: self.shared.limit as u64,
            grown: p.grown,
            grown_ms: p.grown_ms,
        }
    }

    /// An extent for `rows` `u16` and `recurrent` `f32` values, zeroed the
    /// first time its memory was lent. The pool grows by a slab when no slab
    /// has room; refused by name, nothing allocated, when the extent does not
    /// fit the limit beside what the slabs hold.
    pub fn lend(&self, rows: usize, recurrent: usize) -> Result<Staged, GpuError> {
        let layout = Layout::of(rows, recurrent).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("{rows} u16 and {recurrent} f32 values pass usize"),
        })?;
        let mut pool = self.shared.pool();
        let mut alloc_ms = 0.0;
        let at = pool
            .slabs
            .iter_mut()
            .enumerate()
            .find_map(|(i, s)| s.extents.take(layout.total).map(|off| (i, off)));
        let (slab, off) = match at {
            Some(found) => found,
            None => {
                let need = layout.total.next_multiple_of(GRAIN);
                let room = self.shared.limit.saturating_sub(pool.held);
                if need > room {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "a state of {} bytes where the pool of {} bytes holds {} and lends {}",
                            layout.total, self.shared.limit, pool.held, pool.lent
                        ),
                    });
                }
                let size = need.max(SLAB.min(room));
                let t = Instant::now();
                let mem = PinnedHostBuffer::<u8>::zeroed(&self.shared.ctx, size)?;
                alloc_ms = t.elapsed().as_secs_f64() * 1e3;
                let mut extents = Extents::new(size, GRAIN);
                let off = extents
                    .take(layout.total)
                    .expect("a fresh slab holds the extent it was sized for");
                pool.held += size;
                pool.grown += 1;
                pool.grown_ms += alloc_ms;
                pool.slabs.push(Slab { mem, extents });
                (pool.slabs.len() - 1, off)
            }
        };
        let len = layout.total;
        pool.lent += len.next_multiple_of(GRAIN);
        // SAFETY: `off` lies inside the slab (an extent it lent), whose memory
        // lives as long as the pool, which `Staged` keeps alive.
        let base = unsafe { pool.slabs[slab].mem.as_mut_ptr().add(off) };
        Ok(Staged {
            shared: Arc::clone(&self.shared),
            slab,
            off,
            len,
            base,
            rows,
            recurrent,
            layout,
            alloc_ms,
            hold: None,
        })
    }
}

/// An extent of pinned memory holding a state's two host arrays: `rows`
/// `u16` from its start, `recurrent` `f32` from the next cache line
/// ([`Layout`]). The extent goes back to the pool on drop, after the copy
/// that fills it ([`Staged::hold`]) has landed.
pub struct Staged {
    shared: Arc<Shared>,
    slab: usize,
    off: usize,
    len: usize,
    /// The extent's first byte: pinned memory no one else is lent.
    base: *mut u8,
    rows: usize,
    recurrent: usize,
    layout: Layout,
    alloc_ms: f64,
    hold: Option<Arc<CudaEvent>>,
}

// SAFETY: `base` points into a slab the pool owns and keeps alive through
// `shared`, at an extent lent to this value alone; the value is the only
// holder of the pointer, and what it reads or writes through it is plain
// integers and floats.
unsafe impl Send for Staged {}
// SAFETY: a shared reference reads the extent through `&self` methods, which
// the copy in flight never overlaps once its event has been waited for
// ([`Staged::settle`]); writes need `&mut self`.
unsafe impl Sync for Staged {}

impl Staged {
    /// The `u16` rows' count.
    #[must_use]
    pub fn rows_len(&self) -> usize {
        self.rows
    }

    /// The `f32` recurrent values' count.
    #[must_use]
    pub fn recurrent_len(&self) -> usize {
        self.recurrent
    }

    /// The ms the pool spent growing for this extent: 0 when a slab already
    /// had room for it.
    #[must_use]
    pub fn alloc_ms(&self) -> f64 {
        self.alloc_ms
    }

    /// The rows. Read only after the copy that fills them has landed.
    #[must_use]
    pub fn rows(&self) -> &[u16] {
        // SAFETY: `rows` `u16` from the extent's start, inside it (the
        // layout's total counts them) and 2-aligned (the slab's pages and
        // the grain are); no write is in flight past `settle`.
        unsafe { slice::from_raw_parts(self.base.cast::<u16>(), self.rows) }
    }

    /// The recurrent values, as [`Staged::rows`].
    #[must_use]
    pub fn recurrent(&self) -> &[f32] {
        // SAFETY: `recurrent` `f32` from the layout's cache-line boundary,
        // inside the extent and 4-aligned; no write is in flight past
        // `settle`.
        unsafe {
            slice::from_raw_parts(
                self.base.add(self.layout.recurrent_at).cast::<f32>(),
                self.recurrent,
            )
        }
    }

    /// Recurrent values `range`, for the host to write: no copy may write
    /// them (the host-side points of a state, which no copy fills).
    pub fn recurrent_mut(&mut self, range: Range<usize>) -> &mut [f32] {
        assert!(
            range.start <= range.end && range.end <= self.recurrent,
            "recurrent values {range:?} of {}",
            self.recurrent
        );
        // SAFETY: the range lies inside the recurrent array (asserted), whose
        // start is 4-aligned; `&mut self` makes this the only reference, and
        // the device writes the other values only, by the caller's
        // contract.
        unsafe {
            slice::from_raw_parts_mut(
                self.base
                    .add(self.layout.recurrent_at)
                    .cast::<f32>()
                    .add(range.start),
                range.len(),
            )
        }
    }

    /// The first byte of the rows, for a copy to write.
    #[must_use]
    pub fn rows_ptr(&mut self) -> *mut u8 {
        self.base
    }

    /// The bytes the rows take.
    #[must_use]
    pub fn rows_bytes(&self) -> usize {
        self.rows * 2
    }

    /// The first byte of the recurrent values, for a copy to write.
    #[must_use]
    pub fn recurrent_ptr(&mut self) -> *mut u8 {
        // SAFETY: the layout's boundary lies inside the extent.
        unsafe { self.base.add(self.layout.recurrent_at) }
    }

    /// The bytes the recurrent values take.
    #[must_use]
    pub fn recurrent_bytes(&self) -> usize {
        self.recurrent * 4
    }

    /// `event` marks the end of the copy that fills the extent: the extent
    /// does not go back to the pool, and its bytes are not read, before it.
    pub fn hold(&mut self, event: Arc<CudaEvent>) {
        self.hold = Some(event);
    }

    /// Whether a copy still fills the extent.
    pub fn running(&self) -> Result<bool, GpuError> {
        match &self.hold {
            Some(e) => Ok(!e.query()?),
            None => Ok(false),
        }
    }

    /// Waits for the copy that fills the extent, then lets its bytes be
    /// read.
    pub fn settle(&mut self) -> Result<(), GpuError> {
        if let Some(e) = self.hold.take()
            && let Err(err) = e.synchronize()
        {
            self.hold = Some(e);
            return Err(err.into());
        }
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if let Some(e) = self.hold.take() {
            // The copy may still write the extent: wait, or the next state
            // is lent memory a copy is filling. A failed wait is a dead
            // context, which writes nothing more.
            self.shared.ctx.record_err(e.synchronize());
        }
        let mut pool = self.shared.pool();
        pool.slabs[self.slab].extents.give(self.off, self.len);
        pool.lent -= self.len.next_multiple_of(GRAIN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Pitched, dtoh_2d};
    use cuda_core::DeviceBuffer;

    /// A state's memory is cut from a slab the pool grew once: the first
    /// extent pays the allocation, the second pays none and shares the slab,
    /// and the extent a dropped state gives back is lent again at the same
    /// place without another allocation. The two arrays are the extent's own
    /// (a write to one is not seen in the other).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_pool_grows_once_and_lends_the_memory_a_dropped_state_gave_back() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let pool = Staging::new(&ctx, 256 << 20);
        assert_eq!(
            pool.stats().grown,
            0,
            "nothing is allocated until a state asks"
        );
        let mut a = pool.lend(1 << 20, 4096).expect("a first extent");
        assert!(a.alloc_ms() > 0.0, "the first extent grew the pool");
        assert!(a.rows().iter().all(|&v| v == 0) && a.recurrent().iter().all(|&v| v == 0.0));
        a.recurrent_mut(0..4096).fill(1.5);
        let b = pool.lend(1 << 20, 4096).expect("a second extent");
        assert_eq!(b.alloc_ms(), 0.0, "the slab had room");
        assert_eq!(pool.stats().grown, 1, "one slab for both");
        assert!(
            b.rows().iter().all(|&v| v == 0),
            "the second extent is not the first's rows"
        );
        assert!(
            b.recurrent().iter().all(|&v| v == 0.0),
            "nor its recurrent values"
        );
        assert!(a.recurrent().iter().all(|&v| v == 1.5));
        let lent = pool.stats().lent;
        drop(a);
        assert!(pool.stats().lent < lent, "the dropped extent went back");
        let c = pool.lend(1 << 20, 4096).expect("a third extent");
        assert_eq!(
            (c.alloc_ms(), pool.stats().grown),
            (0.0, 1),
            "the given-back memory was lent"
        );
    }

    /// A state that does not fit the limit beside what the slabs hold is
    /// refused by name, nothing allocated; the same request goes through once a
    /// lent extent is given back.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_state_past_the_limit_is_refused_by_name_until_an_extent_comes_back() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let pool = Staging::new(&ctx, 4 << 20);
        // Two extents of 2 MiB fill a pool of 4 MiB.
        let a = pool.lend(1 << 20, 0).expect("a first extent");
        let _b = pool.lend(1 << 20, 0).expect("a second extent");
        let before = pool.stats();
        let err = pool
            .lend(1 << 20, 0)
            .err()
            .expect("a third extent past the pool");
        let msg = err.to_string();
        assert!(msg.contains("staging") && msg.contains("pool of"), "{msg}");
        assert_eq!(
            pool.stats(),
            before,
            "a refusal allocated nothing and lent nothing"
        );
        drop(a);
        let c = pool.lend(1 << 20, 0).expect("the given-back extent");
        assert_eq!((c.alloc_ms(), pool.stats().grown), (0.0, 1));
    }

    /// Dropping an extent whose copy still runs waits for the copy: the
    /// memory is never lent while the card writes it. The read is big enough
    /// that it is still running when the extent is dropped, a few µs after
    /// the call.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_dropping_an_extent_waits_for_the_copy_that_fills_it() {
        const BYTES: usize = 256 << 20;
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let dev = DeviceBuffer::<u8>::zeroed(&stream, BYTES).expect("the device store");
        let pool = Staging::new(&ctx, 2 * BYTES as u64);
        let mut staged = pool.lend(BYTES / 2, 0).expect("an extent");
        let shape = Pitched {
            width: BYTES,
            rows: 1,
            dev_pitch: BYTES,
            host_pitch: BYTES,
        };
        // SAFETY: the extent is the pool's pinned memory, held below until
        // the copy has landed.
        unsafe {
            dtoh_2d(
                &stream,
                &dev,
                0,
                staged.rows_ptr(),
                staged.rows_bytes(),
                shape,
            )
        }
        .expect("a 2-D read");
        let end = Arc::new(ctx.new_event(None).expect("an event"));
        end.record(&stream).expect("recorded");
        staged.hold(Arc::clone(&end));
        assert!(
            staged.running().expect("a query"),
            "the copy had landed before the drop: not a test"
        );
        drop(staged);
        assert!(
            end.query().expect("a query"),
            "the extent went back with its copy still running"
        );
    }
}
