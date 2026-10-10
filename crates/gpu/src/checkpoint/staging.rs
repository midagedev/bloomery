//! Pinned host memory for sequence states saved off the request path
//! ([`Staging`]): page-locked slabs, lent to a state as one extent
//! ([`Staged`]) and taken back when the state is dropped.
//!
//! Page-locking costs far more than a copy over the link, so the pool pins
//! all of its memory once, when it is made ([`Staging::pinned`], at load),
//! and a save never allocates: a state's extent is cut from the slab, and the
//! extent of a state the cache evicted is the next one's. The pool is the
//! prompt cache's budget and no more. A state that finds no free extent gets
//! `None` ([`Staging::lend`]), nothing allocated: the caller makes room
//! (evicts a landed state) or takes the state on the request path. What the
//! pinning cost is recorded ([`StagingStats`]), apart from the copy.
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

impl Pool {
    /// Pins one more slab of `size` bytes and records what it cost: the only
    /// place the pool allocates.
    fn grow(&mut self, ctx: &Arc<CudaContext>, size: usize) -> Result<(), GpuError> {
        let t = Instant::now();
        let mem = PinnedHostBuffer::<u8>::zeroed(ctx, size)?;
        self.grown_ms += t.elapsed().as_secs_f64() * 1e3;
        self.grown += 1;
        self.held += size;
        self.slabs.push(Slab {
            mem,
            extents: Extents::new(size, GRAIN),
        });
        Ok(())
    }
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
    /// A pool of `limit` pinned bytes (rounded down to a whole grain) on
    /// `ctx`'s device, all of it pinned now: the one allocation it ever
    /// makes, at load. A limit under one grain pins nothing, and every
    /// [`Staging::lend`] finds no room. Fails with the driver's error when the
    /// memory cannot be pinned.
    pub fn pinned(ctx: &Arc<CudaContext>, limit: u64) -> Result<Staging, GpuError> {
        let limit = usize::try_from(limit).unwrap_or(usize::MAX) / GRAIN * GRAIN;
        let mut pool = Pool {
            slabs: Vec::new(),
            held: 0,
            lent: 0,
            grown: 0,
            grown_ms: 0.0,
        };
        if limit > 0 {
            pool.grow(ctx, limit)?;
        }
        Ok(Staging {
            shared: Arc::new(Shared {
                ctx: Arc::clone(ctx),
                limit,
                pool: Mutex::new(pool),
            }),
        })
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
    /// first time its memory was lent; `None`, nothing changed, when no free
    /// extent of that size is left (the pool is full of lent extents, or what
    /// is free is in pieces — [`StagingStats::held`] less `lent` tells which).
    pub fn lend(&self, rows: usize, recurrent: usize) -> Result<Option<Staged>, GpuError> {
        let layout = Layout::of(rows, recurrent).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("{rows} u16 and {recurrent} f32 values pass usize"),
        })?;
        let mut pool = self.shared.pool();
        let Some((slab, off)) = pool
            .slabs
            .iter_mut()
            .enumerate()
            .find_map(|(i, s)| s.extents.take(layout.total).map(|off| (i, off)))
        else {
            return Ok(None);
        };
        let len = layout.total;
        pool.lent += len.next_multiple_of(GRAIN);
        // SAFETY: `off` lies inside the slab (an extent it lent), whose memory
        // lives as long as the pool, which `Staged` keeps alive.
        let base = unsafe { pool.slabs[slab].mem.as_mut_ptr().add(off) };
        Ok(Some(Staged {
            shared: Arc::clone(&self.shared),
            slab,
            off,
            len,
            base,
            rows,
            recurrent,
            layout,
            hold: None,
        }))
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

    /// Recurrent values `range`, for the host to write: no copy in flight may
    /// write them.
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

    /// The pool pins its memory once, when it is made, and lends it without
    /// allocating: two extents share the slab, and the extent a dropped state
    /// gives back is lent again at the same place. The two arrays are the
    /// extent's own (a write to one is not seen in the other).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_pool_pins_once_at_load_and_lends_without_allocating() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let pool = Staging::pinned(&ctx, 256 << 20).expect("a pinned pool");
        let at_load = pool.stats();
        assert_eq!(
            (at_load.grown, at_load.held, at_load.lent),
            (1, 256 << 20, 0),
            "the whole limit is pinned at load"
        );
        assert!(at_load.grown_ms > 0.0, "pinning has a cost");
        let mut a = pool
            .lend(1 << 20, 4096)
            .expect("a lend")
            .expect("a first extent");
        assert!(a.rows().iter().all(|&v| v == 0) && a.recurrent().iter().all(|&v| v == 0.0));
        a.recurrent_mut(0..4096).fill(1.5);
        let b = pool
            .lend(1 << 20, 4096)
            .expect("a lend")
            .expect("a second extent");
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
        let _c = pool
            .lend(1 << 20, 4096)
            .expect("a lend")
            .expect("a third extent");
        let after = pool.stats();
        assert_eq!(
            (after.grown, after.grown_ms),
            (at_load.grown, at_load.grown_ms),
            "lending pinned nothing"
        );
    }

    /// A state that fits no free extent gets `None`, nothing allocated or
    /// lent; the same request goes through once a lent extent is given back.
    /// Free space in pieces is no room either: `held - lent` reads as enough
    /// while the lend finds none.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_state_that_fits_no_free_extent_gets_none_until_one_comes_back() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let pool = Staging::pinned(&ctx, 6 << 20).expect("a pinned pool");
        // Three extents of 2 MiB fill a pool of 6 MiB.
        let a = pool
            .lend(1 << 20, 0)
            .expect("a lend")
            .expect("a first extent");
        let b = pool
            .lend(1 << 20, 0)
            .expect("a lend")
            .expect("a second extent");
        let c = pool
            .lend(1 << 20, 0)
            .expect("a lend")
            .expect("a third extent");
        let before = pool.stats();
        assert!(
            pool.lend(1 << 20, 0).expect("a lend").is_none(),
            "a fourth extent past the pool"
        );
        assert_eq!(pool.stats(), before, "a refusal lent nothing");
        drop(b);
        let middle = pool
            .lend(1 << 20, 0)
            .expect("a lend")
            .expect("b's extent again");
        // The ends given back leave 4 MiB free in two pieces of 2 MiB.
        drop(a);
        drop(c);
        let held = pool.stats();
        assert_eq!(held.held - held.lent, 4 << 20);
        assert!(
            pool.lend(2 << 20, 0).expect("a lend").is_none(),
            "4 MiB free, no 4 MiB extent"
        );
        assert_eq!(pool.stats(), held, "a refusal lent nothing");
        drop(middle);
    }

    /// A limit under one grain pins nothing and lends nothing.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_pool_under_one_grain_is_empty() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let pool = Staging::pinned(&ctx, (GRAIN - 1) as u64).expect("an empty pool");
        assert_eq!(pool.stats(), StagingStats::default());
        assert!(pool.lend(1, 0).expect("a lend").is_none());
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
        let pool = Staging::pinned(&ctx, 2 * BYTES as u64).expect("a pinned pool");
        let mut staged = pool.lend(BYTES / 2, 0).expect("a lend").expect("an extent");
        // The stream is held shut, so the copy behind the hold is in flight
        // until this test opens it.
        let hold = crate::checkpoint::hold::Hold::on(&stream).expect("a hold");
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
        let (done, dropped) = std::sync::mpsc::channel();
        let drop_it = std::thread::spawn(move || {
            drop(staged);
            let _ = done.send(());
        });
        assert!(
            dropped
                .recv_timeout(std::time::Duration::from_millis(500))
                .is_err(),
            "the extent went back while its copy was held in flight"
        );
        hold.open();
        drop_it.join().expect("the drop");
        assert!(
            end.query().expect("a query"),
            "the extent went back with its copy still running"
        );
    }
}
